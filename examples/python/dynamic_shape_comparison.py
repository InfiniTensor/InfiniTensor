"""Reproducible CPU attention comparisons with raw samples and ORT checks.

Each pair changes only folding or trimming. Times include Python overhead;
end-to-end spans set_input through output copy, including trim when enabled.
Run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from dynamic_shape_attention import attention_input
from dynamic_shape_benchmark import declared_shapes
from pyinfinitensor.onnx import OnnxStub, backend


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_stub(model_bytes, shapes, pin_seq, fold):
    stub = OnnxStub(onnx.load_model_from_string(model_bytes), backend.cpu_runtime())
    stub.set_input([list(shapes[0])])
    if pin_seq is not None:
        stub.pin_dims("x", [1])
    before = [len(stub.handler.operators()), stub.shape_subgraph_size()]
    dropped = stub.fold_shape_subgraph() if fold else 0
    after = [len(stub.handler.operators()), stub.shape_subgraph_size()]
    if fold:
        assert dropped > 0 and before[0] - after[0] == dropped
        assert stub.fold_shape_subgraph() == 0
    return stub, dict(
        nodes_before=before[0],
        shape_nodes_before=before[1],
        nodes=after[0],
        shape_nodes=after[1],
        dropped=dropped,
    )


def serve(stub, shape, data, trim):
    stub.set_input([shape])
    if trim:
        stub.trim_memory()
    stub.inputs["x"].copyin_numpy(data)
    stub.run()
    return stub.outputs["y"].copyout_numpy()


def measure(stub, shapes, arrays, trim, warmup, repeats):
    for _ in range(warmup):
        for shape, data in zip(shapes, arrays):
            serve(stub, shape, data, trim)
    previous = stub.inputs["x"].shape()
    rows = []
    for repeat in range(repeats):
        for step, (shape, data) in enumerate(zip(shapes, arrays)):
            changed = shape != previous
            allocations_before = stub.activation_allocations()
            t0 = time.perf_counter_ns()
            stub.set_input([shape])
            t1 = time.perf_counter_ns()
            if trim:
                stub.trim_memory()
                t2 = time.perf_counter_ns()
            else:
                t2 = t1
            stub.inputs["x"].copyin_numpy(data)
            t3 = time.perf_counter_ns()
            stub.run()
            t4 = time.perf_counter_ns()
            output = stub.outputs["y"].copyout_numpy()
            t5 = time.perf_counter_ns()
            assert list(output.shape) == shape
            rows.append(
                dict(
                    repeat=repeat,
                    step=step,
                    batch=shape[0],
                    seq=shape[1],
                    changed=changed,
                    set_input_ms=(t1 - t0) / 1e6,
                    trim_ms=(t2 - t1) / 1e6,
                    copyin_ms=(t3 - t2) / 1e6,
                    run_ms=(t4 - t3) / 1e6,
                    copyout_ms=(t5 - t4) / 1e6,
                    e2e_ms=(t5 - t0) / 1e6,
                    allocations=stub.activation_allocations() - allocations_before,
                    allocated_bytes=stub.allocated_bytes(),
                    activation_capacity=stub.activation_capacity(),
                    activation_peak=stub.activation_peak(),
                )
            )
            previous = shape
    return rows


METRICS = ("set_input_ms", "trim_ms", "copyin_ms", "run_ms", "copyout_ms", "e2e_ms")


def summarize_trial(rows):
    result = {key: statistics.median(row[key] for row in rows) for key in METRICS}
    for changed in (True, False):
        subset = [row["set_input_ms"] for row in rows if row["changed"] == changed]
        label = "changed" if changed else "same"
        result["set_input_" + label + "_ms"] = (
            statistics.median(subset) if subset else None
        )
        result[label + "_samples"] = len(subset)
    result.update(
        allocations=sum(row["allocations"] for row in rows),
        mean_held_bytes=statistics.mean(row["allocated_bytes"] for row in rows),
        peak_held_bytes=max(row["allocated_bytes"] for row in rows),
        mean_capacity_bytes=statistics.mean(row["activation_capacity"] for row in rows),
        peak_capacity_bytes=max(row["activation_capacity"] for row in rows),
    )
    return result


def spread(values):
    return dict(median=statistics.median(values), min=min(values), max=max(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if min(args.trials, args.repeat, args.warmup) < 1:
        parser.error("trial, repeat and warmup counts must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    # One logical CPU avoids migration; it does not provide exclusive CPU use.
    original_affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {original_affinity[0]})
    model_bytes = args.model.read_bytes()
    model = onnx.load_model_from_string(model_bytes)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    reference = ort.InferenceSession(
        model_bytes, options, providers=["CPUExecutionProvider"]
    )
    scenarios = [
        (
            "fold_seq5",
            [(b, 5) for b in (2, 1, 8, 3, 2)],
            5,
            [("plain", False, False), ("fold", True, False)],
        ),
        (
            "fold_seq64",
            [(b, 64) for b in (2, 1, 8, 3, 2)],
            64,
            [("plain", False, False), ("fold", True, False)],
        ),
        (
            "capacity_short",
            [(2, 5), (1, 1), (2, 7), (3, 16), (1, 3), (2, 5)],
            None,
            [("trim", False, True), ("reuse", False, False)],
        ),
        (
            "capacity_large",
            [(2, 16), (1, 64), (2, 128), (3, 32), (1, 8), (2, 16)],
            None,
            [("trim", False, True), ("reuse", False, False)],
        ),
    ]
    repo = Path(__file__).resolve().parents[2]
    metadata = dict(
        model_sha256=sha256(args.model),
        declared_shapes=declared_shapes(model),
        head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        tracked_diff_sha256=hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=repo)
        ).hexdigest(),
        script_sha256=sha256(__file__),
        backend_path=str(Path(backend.__file__).resolve()),
        backend_sha256=sha256(Path(backend.__file__).resolve()),
        lib_sha256=sha256(repo / "build/Release/libInfiniTensor.so"),
        python=platform.python_version(),
        platform=platform.platform(),
        versions={
            name: importlib.metadata.version(name)
            for name in ("numpy", "onnx", "onnxsim", "onnxruntime", "torch")
        },
        cpuinfo=Path("/proc/cpuinfo").read_text(),
        original_affinity=original_affinity,
        measured_affinity=sorted(os.sched_getaffinity(0)),
        threads={
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        trials=args.trials,
        repeats=args.repeat,
        warmup_sequences=args.warmup,
        simplify="default",
        input_seed="[2026,9,7,batch,seq]",
        timer="perf_counter_ns",
        e2e="set_input + trim if enabled + input copy + run + output copy; no import, RNG, ORT or counters",
        memory="allocator-held bytes after inference, not process RSS; trim after set_input before input copy",
        independence="fresh graph per strategy per trial, alternating order; shared process/runtime/OS caches",
    )
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    validation, prepared = [], []
    # Every requested shape/policy must pass before any reported measurement.
    for name, pairs, pin, policies in scenarios:
        shapes = [[b, s, 32] for b, s in pairs]
        arrays = [attention_input(b, s) for b, s in pairs]
        wants = [reference.run(["y"], {"x": data})[0] for data in arrays]
        first_outputs = None
        for label, fold, trim in policies:
            stub, nodes = make_stub(model_bytes, shapes, pin, fold)
            outputs = []
            for shape, data, want in zip(shapes, arrays, wants):
                got = serve(stub, shape, data, trim)
                np.testing.assert_allclose(
                    got, want, rtol=1e-4, atol=1e-5, equal_nan=False
                )
                outputs.append(got)
                validation.append(
                    dict(
                        scenario=name,
                        policy=label,
                        shape=shape,
                        max_abs_ort_error=float(np.max(np.abs(got - want))),
                        **nodes
                    )
                )
            if first_outputs is not None:
                for left, right in zip(first_outputs, outputs):
                    np.testing.assert_array_equal(left, right)
            first_outputs = outputs
            del stub
        prepared.append((name, shapes, arrays, pin, policies))
    (args.out / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(
        "VALIDATED",
        len(validation),
        "ORT comparisons; all policy pairs exactly equal",
        flush=True,
    )

    trials, summary = [], {}
    with (args.out / "samples.csv").open("w", newline="") as stream:
        writer = None
        for name, shapes, arrays, pin, policies in prepared:
            for trial in range(args.trials):
                ordered = policies if trial % 2 == 0 else policies[::-1]
                for order, (label, fold, trim) in enumerate(ordered):
                    stub, nodes = make_stub(model_bytes, shapes, pin, fold)
                    rows = measure(stub, shapes, arrays, trim, args.warmup, args.repeat)
                    for row in rows:
                        row.update(
                            scenario=name, trial=trial, order=order, policy=label
                        )
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        writer.writeheader()
                    writer.writerows(rows)
                    stream.flush()
                    stats = dict(
                        scenario=name,
                        trial=trial,
                        order=order,
                        policy=label,
                        **nodes,
                        **summarize_trial(rows)
                    )
                    trials.append(stats)
                    print("TRIAL " + json.dumps(stats, sort_keys=True), flush=True)
                    del stub
            base, optimized = policies[0][0], policies[1][0]
            subset = [t for t in trials if t["scenario"] == name]
            summary[name] = {
                label: {
                    key: spread([t[key] for t in subset if t["policy"] == label])
                    for key in summarize_trial(rows)
                    if rows and subset[0][key] is not None
                }
                for label, _, _ in policies
            }
            summary[name]["optimized_over_baseline"] = {
                key: spread(
                    [
                        next(
                            t[key]
                            for t in subset
                            if t["trial"] == i and t["policy"] == optimized
                        )
                        / next(
                            t[key]
                            for t in subset
                            if t["trial"] == i and t["policy"] == base
                        )
                        for i in range(args.trials)
                    ]
                )
                for key in ("set_input_ms", "run_ms", "e2e_ms", "mean_held_bytes")
            }
    (args.out / "trials.json").write_text(json.dumps(trials, indent=2) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
