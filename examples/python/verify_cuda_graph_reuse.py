"""Whether a CUDA Graph survives a change of shape, and what it buys.

A CUDA Graph records a launch sequence once and replays it, which is worth
having when a model is run over and over. A dynamic shape is the awkward case:
the recording holds the addresses and sizes that were current when it was made,
so a graph recorded for one shape must not be replayed for another. The four
things asked here are, in order:

  capture      a first run records a graph and the answer is still right
  invalidate   a change of shape does not replay the stale recording
  recapture    the new shape records a graph of its own
  reuse        returning to a shape already seen replays rather than records

The fourth is the one that matters for a serving loop, where a handful of shapes
recur. `cuda_graph_capture_count` answers all four: it rises only when a
recording is made, so reuse shows up as a run that does not raise it.

Correctness is checked against onnxruntime at every step, because a graph that
replays the wrong recording would otherwise look like a fast success. The
numbers are compared, not just the shapes.

Then the benefit: the same shape run many times, with and without the graph.
The gain is in launch overhead, so it shows on small shapes and shrinks as the
kernels grow -- which is itself worth reporting rather than hiding behind one
average.

Written to be run on the card:

    python3 verify_cuda_direction4.py --models /data/models_pkg/18_shape_map_after --out /data/d4
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from pyinfinitensor.onnx import OnnxStub, backend

parser = argparse.ArgumentParser()
parser.add_argument("--models", required=True, help="directory holding the .onnx files")
parser.add_argument("--out", required=True, help="directory to write results into")
parser.add_argument("--rtol", type=float, default=1e-4)
parser.add_argument("--atol", type=float, default=1e-5)
parser.add_argument("--repeats", type=int, default=200,
                    help="timed runs per shape, after warmup")
args = parser.parse_args()

models_dir = Path(args.models)
out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

# A shape is revisited deliberately: (2,5,32) opens and closes the attention
# sequence, and (2,3,48,48) does the same for resnet. The last entry is what
# tests reuse -- by then its recording is several shapes old, so a cache that
# only remembered the previous shape would fail here.
CASES = [
    (
        "attention",
        "attention.onnx",
        "x",
        [(2, 5, 32), (1, 1, 32), (2, 7, 32), (2, 5, 32)],
    ),
    (
        "resnet18",
        "resnet18.onnx",
        "images",
        [(2, 3, 48, 48), (1, 3, 32, 32), (4, 3, 64, 64), (2, 3, 48, 48)],
    ),
]


def feed_for(shape, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=shape, dtype=np.float32)


def run_ort(session, input_name, data):
    return session.run(None, {input_name: data})


def set_and_run(stub, shape, data, use_graph):
    """Put one shape through the stub, the way a caller would.

    `set_input` decides for itself what a shape costs: a shape already in place
    returns at once, and only a changed one infers shapes and lays out memory
    again. Calling shape_infer and data_malloc here as well would lay the memory
    out on every run, including the runs where nothing changed -- and since a
    tensor's address is part of what a recorded graph is matched on, that would
    move every tensor and force a fresh recording each time. The reuse this
    whole script is meant to observe would then never happen, and the failure
    would be one the script had manufactured. So the stub is left to do it.
    """
    stub.set_input([list(shape)])
    for tensor in stub.inputs.values():
        tensor.copyin_numpy(data)
    if use_graph:
        stub.run_with_cudagraph()
    else:
        stub.run()
    return [np.array(t.copyout_numpy()) for t in stub.outputs.values()]


def compare(ours, theirs, rtol, atol):
    """The largest disagreement across every output, or None if the shapes differ."""
    if len(ours) != len(theirs):
        return None, "output count %d vs %d" % (len(ours), len(theirs))
    worst = 0.0
    for i, (a, b) in enumerate(zip(ours, theirs)):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.shape != b.shape:
            return None, "output %d shape %s vs %s" % (i, a.shape, b.shape)
        worst = max(worst, float(np.max(np.abs(a - b))))
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            return worst, "output %d outside tolerance" % i
    return worst, None


def four_properties(name, path, input_name, shapes):
    """Capture, invalidate, recapture and reuse, on one model.

    The runtime is asked for its capture count around every run. What the count
    does -- rise or hold -- is the finding; the shape sequence is arranged so
    that each of the four shows up as a specific pattern in it.
    """
    record = {"model": name, "ok": False, "error": None, "runs": []}
    try:
        model = onnx.load(str(path))
        session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"])

        runtime = backend.cuda_runtime()
        stub = OnnxStub(model, runtime)

        seen = {}
        for step, shape in enumerate(shapes):
            before = runtime.cuda_graph_capture_count()
            data = feed_for(shape, seed=1000 + step)
            ours = set_and_run(stub, shape, data, use_graph=True)
            after = runtime.cuda_graph_capture_count()

            theirs = run_ort(session, input_name, data)
            worst, complaint = compare(ours, theirs, args.rtol, args.atol)

            captured = after > before
            first_time = shape not in seen
            entry = {
                "step": step,
                "shape": list(shape),
                "first_time": first_time,
                "captured": captured,
                "capture_count": after,
                "cache_size": runtime.cuda_graph_cache_size(),
                "max_abs_error": worst,
                "mismatch": complaint,
            }

            # A shape not seen before has to record -- there is nothing to
            # replay. Whether a shape seen before replays is the open question,
            # not a requirement to assert here: the cache keys on tensor
            # addresses among other things, and a shape change lays memory out
            # again, so whether the addresses come back to what they were is a
            # property of the allocator that this run is here to find out. So
            # only the first half is called expected; the second is recorded.
            entry["as_expected"] = captured if first_time else None
            record["runs"].append(entry)
            seen[shape] = True

            if complaint is not None:
                record["error"] = "step %d: %s" % (step, complaint)
                return record

        # A step whose expectation was settled beforehand and came out wrong.
        # A repeat step is not among these: what it does is the measurement.
        wrong = [r for r in record["runs"] if r["as_expected"] is False]
        if wrong:
            record["error"] = "capture behaviour wrong at steps %s" % (
                [r["step"] for r in wrong],)
            return record

        # Name each property against the runs that showed it, so the result says
        # which question each answered rather than leaving it to be read off the
        # table.
        firsts = [r for r in record["runs"] if r["first_time"]]
        repeats = [r for r in record["runs"] if not r["first_time"]]
        record["capture"] = bool(firsts and firsts[0]["captured"])
        record["invalidate_and_recapture"] = bool(
            len(firsts) > 1 and all(r["captured"] for r in firsts[1:]))
        record["distinct_shapes"] = len(firsts)
        record["max_abs_error"] = max(
            (r["max_abs_error"] or 0.0) for r in record["runs"])

        # Whether a shape seen before is replayed from its recording or recorded
        # again. This is the measurement, not a requirement: laying out memory
        # again is what a shape change costs, and it is also what moves tensors,
        # while a recording is keyed partly on where its tensors sat. So a
        # returning shape finds its recording only if the addresses came back --
        # which is a property of the allocator, and is what this reports.
        record["reuse"] = bool(repeats and not any(r["captured"] for r in repeats))
        record["repeat_steps"] = len(repeats)
        record["reuse_note"] = (
            "a returning shape replayed its recording"
            if record["reuse"] else
            "a returning shape recorded again -- the cache does not span a "
            "shape change and forth, so it pays only within a run of one shape")

        # Correctness always, and the two properties whose failure would mean
        # the feature does nothing or does the wrong thing. Reuse is reported
        # rather than required.
        record["ok"] = (record["capture"]
                        and record["invalidate_and_recapture"])
        if not record["ok"]:
            record["error"] = (
                "nothing was ever recorded" if not record["capture"]
                else "a new shape did not record afresh")
    except Exception as exc:  # noqa: BLE001 -- the message is the finding
        record["error"] = "%s: %s" % (type(exc).__name__, exc)
    return record


def latency(name, path, shapes):
    """What replaying a recording saves, per shape.

    Both arms run the same shape the same number of times on the same stub, so
    what differs is only whether the launches were replayed from a recording.
    The graph arm is measured after its first run, so the recording itself is
    not counted as part of what replaying costs.
    """
    record = {"model": name, "ok": False, "error": None, "shapes": []}
    try:
        model = onnx.load(str(path))
        runtime = backend.cuda_runtime()
        stub = OnnxStub(model, runtime)

        for shape in shapes:
            data = feed_for(shape, seed=7)

            def timed(use_graph, n):
                # Warm up, so neither arm pays for a first-time cost the other
                # does not: the graph arm records here, the plain arm settles
                # whatever its own first run settles.
                #
                # Nothing syncs the device explicitly, because there is no
                # binding that does -- CudaRuntime exposes only the cache
                # accessors and init_comm. It is not needed either: both
                # runWithCudaGraph and run sync before returning, and
                # `set_and_run` copies the outputs back to the host, which
                # cannot complete before the work that produced them.
                for _ in range(5):
                    set_and_run(stub, shape, data, use_graph)
                start = time.perf_counter()
                for _ in range(n):
                    set_and_run(stub, shape, data, use_graph)
                return (time.perf_counter() - start) / n * 1e3

            plain_ms = timed(False, args.repeats)
            graph_ms = timed(True, args.repeats)
            record["shapes"].append({
                "shape": list(shape),
                "elements": int(np.prod(shape)),
                "plain_ms": round(plain_ms, 4),
                "graph_ms": round(graph_ms, 4),
                "speedup": round(plain_ms / graph_ms, 3) if graph_ms else None,
            })
        record["ok"] = True
    except Exception as exc:  # noqa: BLE001
        record["error"] = "%s: %s" % (type(exc).__name__, exc)
    return record


results = {"properties": [], "latency": []}
failed = 0

for name, filename, input_name, shapes in CASES:
    path = models_dir / filename
    if not path.exists():
        print("D4 missing model: %s" % path)
        failed += 1
        continue

    prop = four_properties(name, path, input_name, shapes)
    results["properties"].append(prop)
    print("D4 " + json.dumps(prop))
    if not prop["ok"]:
        failed += 1
        continue

    # Timing a model whose capture behaviour is wrong would only mislead.
    lat = latency(name, path, sorted(set(shapes)))
    results["latency"].append(lat)
    print("D4_LATENCY " + json.dumps(lat))
    if not lat["ok"]:
        failed += 1

(out_dir / "direction4.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8")

print("D4_COMPLETE models=%d failed=%d" % (len(CASES), failed))
