"""Measure dynamic-shape preparation and execution latency on CPU.

Run from the repository root after building/installing the Python backend:
    python3 scripts/benchmark_dynamic_shape.py
"""

import argparse
import statistics
import time

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from pyinfinitensor import backend
from pyinfinitensor.onnx import OnnxStub


def make_model():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 6])
    nodes = [
        helper.make_node("Shape", ["x"], ["shape"]),
        helper.make_node(
            "Gather", ["shape", "index"], ["batch"], axis=0
        ),
        helper.make_node("Unsqueeze", ["batch", "axes"], ["batch_vector"]),
        helper.make_node("Concat", ["batch_vector", "tail"], ["target"], axis=0),
        helper.make_node("Reshape", ["x", "target"], ["y"]),
    ]
    initializers = [
        numpy_helper.from_array(np.array(0, dtype=np.int64), name="index"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="axes"),
        numpy_helper.from_array(np.array([6], dtype=np.int64), name="tail"),
    ]
    return helper.make_model(
        helper.make_graph(nodes, "dynamic_shape_benchmark", [x], [y], initializers),
        opset_imports=[helper.make_opsetid("", 18)],
    )


def percentile(values, q):
    values = sorted(values)
    index = min(len(values) - 1, int(round((q / 100) * (len(values) - 1))))
    return values[index]


def run_case(stub, shapes, warmup, repeat):
    for batch in shapes[:warmup]:
        values = np.arange(batch * 6, dtype=np.float32).reshape(batch, 2, 3)
        stub.set_input([[batch, 2, 3]])
        stub.inputs["x"].copyin_numpy(values)
        stub.run()

    samples = []
    for batch in shapes[warmup : warmup + repeat]:
        values = np.arange(batch * 6, dtype=np.float32).reshape(batch, 2, 3)
        start = time.perf_counter_ns()
        stub.set_input([[batch, 2, 3]])
        stub.inputs["x"].copyin_numpy(values)
        prepared = time.perf_counter_ns()
        stub.run()
        finished = time.perf_counter_ns()
        samples.append(((prepared - start) / 1e6, (finished - start) / 1e6))
    prep = [item[0] for item in samples]
    total = [item[1] for item in samples]
    return prep, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--workspace-size", type=int, default=64 << 20)
    args = parser.parse_args()
    model = make_model()
    if args.device == "cuda":
        runtime = backend.cuda_runtime(workspace_size=args.workspace_size)
    else:
        runtime = backend.cpu_runtime()
    for name, batches in {
        "same_shape": [8] * (args.warmup + args.repeat),
        "alternating_shapes": [1, 8, 2, 5] * ((args.warmup + args.repeat + 3) // 4),
    }.items():
        stub = OnnxStub(model, runtime)
        prep, total = run_case(stub, batches, args.warmup, args.repeat)
        print(
            f"{name}: prep_ms p50={percentile(prep, 50):.4f} "
            f"p99={percentile(prep, 99):.4f}; "
            f"total_ms p50={percentile(total, 50):.4f} "
            f"p99={percentile(total, 99):.4f}; "
            f"prep_mean={statistics.mean(prep):.4f}"
        )


if __name__ == "__main__":
    main()
