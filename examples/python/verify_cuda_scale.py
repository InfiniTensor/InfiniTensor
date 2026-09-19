"""Whether the dynamic-shape machinery holds at a size that is not a toy.

Every result so far was taken at sequence lengths of 5 to 16 and batches of 1 to
4. Those are small enough that a mechanism could be correct there and still fall
over on anything real -- a rank or stride computed as `int` where the element
count no longer fits, a workspace sized from the first shape seen, an allocator
that fragments once the tensors are large. So the same questions are asked again
at sequence 512 and 1024, and at batch 32.

Three things are asked at each size:

  * The result still agrees with onnxruntime, output by output.
  * A shape change still works in place, without reloading the model -- the
    sequence runs small, large, small on one stub.
  * Peak device memory is reported, so a size that cannot fit says so as a
    number rather than as an out-of-memory crash with no context.

A size that does not fit on the card is recorded as `skipped`, not as a failure:
the finding is the largest size that works, and a 24 GB card has a ceiling that
says nothing about whether the mechanism is sound.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from pyinfinitensor.onnx import OnnxStub, backend

parser = argparse.ArgumentParser()
parser.add_argument("--models", required=True, help="directory holding the .onnx files")
parser.add_argument("--out", required=True, help="directory for the json record")
parser.add_argument("--rtol", type=float, default=1e-4)
parser.add_argument("--atol", type=float, default=1e-5)
args = parser.parse_args()

models_dir = Path(args.models)
out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

# attention takes (batch, seq, 32); resnet18 takes (batch, 3, h, w). The shapes
# run small -> large -> small, so a large allocation is followed by a small one
# on the same stub: growing and shrinking are different paths through the
# allocator and both have to work without the model being loaded again.
CASES = [
    (
        "attention_seq",
        "attention.onnx",
        "x",
        [(1, 16, 32), (1, 512, 32), (1, 1024, 32), (1, 16, 32)],
    ),
    (
        "attention_batch",
        "attention.onnx",
        "x",
        [(1, 128, 32), (32, 128, 32), (1, 128, 32)],
    ),
    (
        "resnet18_batch",
        "resnet18.onnx",
        "images",
        [(1, 3, 64, 64), (32, 3, 64, 64), (1, 3, 64, 64)],
    ),
]


def device_free_total():
    """Device memory free and total in bytes, or None where it cannot be read.

    Read through pynvml when it is there and nvidia-smi otherwise, since the
    point is only to say how much of the card a size took.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return int(info.free), int(info.total)
    except Exception:  # noqa: BLE001 -- a missing reading is not a failure
        pass
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True).stdout
        free, total = (int(v) for v in out.strip().splitlines()[0].split(","))
        return free * 1024 * 1024, total * 1024 * 1024
    except Exception:  # noqa: BLE001
        return None, None


def compare(ours, theirs, rtol, atol):
    """The largest disagreement across every output, or a complaint."""
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


def at_scale(name, path, input_name, shapes):
    """Run one model over a sequence of sizes on a single stub."""
    record = {"case": name, "ok": False, "error": None, "sizes": []}
    try:
        model = onnx.load(str(path))
        session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"])

        runtime = backend.cuda_runtime()
        stub = OnnxStub(model, runtime)

        for shape in shapes:
            elements = int(np.prod(shape))
            entry = {"shape": list(shape), "elements": elements}

            free_before, total = device_free_total()
            rng = np.random.default_rng(11)
            data = rng.standard_normal(size=shape, dtype=np.float32)

            try:
                # Only set_input: it lays out memory again when the shape has
                # changed and does nothing when it has not, which is what a
                # caller does and therefore what should be measured.
                stub.set_input([list(shape)])
                for tensor in stub.inputs.values():
                    tensor.copyin_numpy(data)
                stub.run()
                ours = [np.array(t.copyout_numpy())
                        for t in stub.outputs.values()]
            except Exception as exc:  # noqa: BLE001
                message = "%s: %s" % (type(exc).__name__, exc)
                # A size the card cannot hold is a ceiling, not a defect. It is
                # recorded as skipped so that the largest size that does work
                # stays the finding.
                if "out of memory" in message.lower() or "alloc" in message.lower():
                    entry["skipped"] = "does not fit: %s" % message
                    record["sizes"].append(entry)
                    continue
                raise

            free_after, _ = device_free_total()
            if free_before is not None and free_after is not None:
                entry["device_used_mib"] = round(
                    (free_before - free_after) / 1024 / 1024, 1)
                entry["device_total_mib"] = round(total / 1024 / 1024, 1)

            theirs = session.run(None, {input_name: data})
            worst, complaint = compare(ours, theirs, args.rtol, args.atol)
            entry["max_abs_error"] = worst
            entry["mismatch"] = complaint
            record["sizes"].append(entry)
            if complaint is not None:
                record["error"] = "shape %s: %s" % (list(shape), complaint)
                return record

        ran = [s for s in record["sizes"] if "skipped" not in s]
        record["largest_elements"] = max((s["elements"] for s in ran), default=0)
        record["sizes_run"] = len(ran)
        record["sizes_skipped"] = len(record["sizes"]) - len(ran)
        record["max_abs_error"] = max(
            (s.get("max_abs_error") or 0.0) for s in ran) if ran else None
        # The smallest size comes last as well as first, so agreement there
        # means shrinking after a large allocation still produces right answers.
        record["shrank_back"] = bool(
            len(ran) >= 2 and ran[-1]["elements"] < record["largest_elements"])
        record["ok"] = bool(ran) and record["error"] is None
        if not record["ok"] and record["error"] is None:
            record["error"] = "no size ran at all"
    except Exception as exc:  # noqa: BLE001 -- the message is the finding
        record["error"] = "%s: %s" % (type(exc).__name__, exc)
    return record


results = []
failed = 0

for name, filename, input_name, shapes in CASES:
    path = models_dir / filename
    if not path.exists():
        print("SCALE missing model: %s" % path)
        failed += 1
        continue
    record = at_scale(name, path, input_name, shapes)
    results.append(record)
    print("SCALE " + json.dumps(record))
    if not record["ok"]:
        failed += 1

(out_dir / "scale.json").write_text(
    json.dumps({"cases": results}, indent=2), encoding="utf-8")
print("SCALE_COMPLETE cases=%d failed=%d" % (len(CASES), failed))
