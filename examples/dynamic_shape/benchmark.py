"""Compare constant folding and pool reuse with synchronized ORT-checked runs."""

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import onnxruntime as ort
from pyinfinitensor import backend
from pyinfinitensor.onnx import OnnxStub
from models import batch_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--features", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=Path("benchmark.json"))
    args = parser.parse_args()
    if args.repeats < 1 or (args.cuda_graph and args.device != "cuda"):
        parser.error("positive repeats and CUDA device for CUDA Graph are required")
    model = batch_model(args.features, 16)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    reference = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(2026)
    inputs = [
        rng.normal(size=(b, 2, args.features // 2)).astype(np.float32)
        for b in [1, 2, 8, 3, 1]
    ]
    expected = [reference.run(None, {"x": x})[0] for x in inputs]
    results = []
    for folding, reuse in [(False, False), (True, False), (False, True), (True, True)]:
        runtime = (
            backend.cpu_runtime() if args.device == "cpu" else backend.cuda_runtime()
        )
        stub = OnnxStub(model, runtime, optimize_shapes=folding, memory_reuse=reuse)
        samples = []
        for repeat in range(args.repeats + 1):
            for x, y in zip(inputs, expected):
                start = time.perf_counter_ns()
                stub.set_input([x.shape])
                prepared = time.perf_counter_ns()
                stub.inputs["x"].copyin_numpy(x)
                uploaded = time.perf_counter_ns()
                if args.cuda_graph:
                    stub.run_with_cudagraph()
                else:
                    stub.run()
                executed = time.perf_counter_ns()
                actual = stub.outputs["y"].copyout_numpy()
                finished = time.perf_counter_ns()
                np.testing.assert_allclose(actual, y, rtol=1e-4, atol=1e-5)
                if repeat:
                    samples.append(
                        {
                            "prepare_ms": (prepared - start) / 1e6,
                            "execute_ms": (executed - uploaded) / 1e6,
                            "end_to_end_ms": (finished - start) / 1e6,
                        }
                    )
            if repeat == 0:
                warmup_allocations = stub.memory_stats()["activation_allocations"]
        summary = {
            key: {
                "mean": float(np.mean([s[key] for s in samples])),
                "median": float(np.median([s[key] for s in samples])),
                "p95": float(np.percentile([s[key] for s in samples], 95)),
            }
            for key in samples[0]
        }
        record = {
            "constant_folding": folding,
            "capacity_reuse": reuse,
            "shape_optimization": stub.shape_optimization,
            "runtime_shape_nodes": stub.handler.shape_compute_count(),
            "timings": summary,
            "memory": stub.memory_stats(),
            "timed_pool_allocations": stub.memory_stats()["activation_allocations"]
            - warmup_allocations,
        }
        if args.cuda_graph:
            record["cuda_graph_captures"] = runtime.cuda_graph_capture_count()
        results.append(record)
        print(json.dumps(record))
        # The existing CUDA runtime reserves a large workspace. Release it
        # before constructing the next independent benchmark configuration.
        del stub
        del runtime
    payload = {
        "device": args.device,
        "cuda_graph": args.cuda_graph,
        "features": args.features,
        "sequence": [1, 2, 8, 3, 1],
        "warmup_sequences": 1,
        "timed_sequences": args.repeats,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "timing_scope": "prepare includes shape evaluation and memory planning; end-to-end includes input/output copies; each run is synchronized",
        "memory_scope": "activation pool capacity and old/new pool overlap during reallocation; excludes runtime workspace, library metadata, Python/ORT and externally retained tensor views",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
