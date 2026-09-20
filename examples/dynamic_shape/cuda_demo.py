"""Criterion 3: persistent CUDA dynamic shapes, independently checked on CPU/ORT.

The shape control program runs on the host. All float data operators run on
InfiniTensor CUDA. This script fails, rather than skips, without a CUDA build.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort

from demo import model
from pyinfinitensor import backend
from pyinfinitensor.onnx import DynamicOnnxStub


IMAGE_SHAPES = [
    (1, 3, 8, 8),
    (2, 3, 12, 10),
    (8, 3, 64, 48),
    (3, 3, 10, 8),
    (1, 3, 8, 8),
]
SEQUENCE_SHAPES = [(1, 3, 4), (2, 8, 4), (8, 64, 4), (3, 5, 4), (1, 3, 4)]


def validate(sequence, runtime, *, naive=False):
    m = model(sequence)
    cuda = DynamicOnnxStub(m, runtime, use_naive_allocator=naive)
    cpu = DynamicOnnxStub(m, backend.cpu_runtime())
    reference = ort.InferenceSession(
        m.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    shapes = SEQUENCE_SHAPES if sequence else IMAGE_SHAPES
    rng = np.random.default_rng(2026)
    rows, tensor_ids, operator_ids = [], None, None
    weight_name = "weight" if sequence else "bias"
    saved_weight = None
    for shape in shapes:
        x = rng.standard_normal(shape).astype(np.float32)
        expected = reference.run(None, {"x": x})[0]
        cpu_result = cpu.run({"x": x})["y"]
        start = time.perf_counter()
        actual = cuda.run({"x": x})["y"]
        elapsed_ms = (time.perf_counter() - start) * 1000
        np.testing.assert_equal(actual.shape, expected.shape)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(actual, cpu_result, rtol=1e-4, atol=1e-5)
        current_ids = tuple(t.fuid() for t in cuda.tensors.values())
        current_ops = tuple(id(op) for op in cuda.ops.values())
        if tensor_ids is None:
            tensor_ids, operator_ids = current_ids, current_ops
            saved_weight = cuda.tensors[weight_name].copyout_numpy()
        assert tensor_ids == current_ids, "CUDA tensors were replaced"
        assert operator_ids == current_ops, "CUDA operators were rebuilt"
        np.testing.assert_array_equal(
            cuda.tensors[weight_name].copyout_numpy(), saved_weight
        )
        row = {
            "input": list(shape),
            "output": list(actual.shape),
            "max_abs_error_ort": float(np.max(np.abs(actual - expected))),
            "max_abs_error_cpu": float(np.max(np.abs(actual - cpu_result))),
            "end_to_end_ms": elapsed_ms,
        }
        if not naive:
            row["memory"] = cuda.handler.memory_stats()
        rows.append(row)
    if not naive:
        memories = [row["memory"] for row in rows]
        assert (
            memories[2]["activation_allocations"]
            > memories[0]["activation_allocations"]
        )
        assert (
            memories[2]["activation_allocations"]
            == memories[4]["activation_allocations"]
        )
        assert (
            memories[4]["activation_capacity_bytes"]
            > memories[4]["activation_required_bytes"]
        )
        for memory in memories:
            assert (
                memory["activation_capacity_bytes"]
                >= memory["activation_required_bytes"]
            )
    return {
        "runs": rows,
        "stats": cuda.stats,
        "persistent_tensors_and_operators": True,
        "weights_preserved": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/dynamic_shape/cuda-results.json")
    parser.add_argument("--naive", action="store_true")
    args = parser.parse_args()
    if not hasattr(backend, "CudaRuntime"):
        raise RuntimeError(
            "Build and install the CUDA backend before running this demo"
        )
    runtime = backend.CudaRuntime(device=0, workspace_size=64 * 1024 * 1024)
    result = {
        "backend": "InfiniTensor CUDA",
        "shape_control": "host",
        "cuda_graph": False,
        "allocator": "naive" if args.naive else "pooled",
        "workspace_bytes": runtime.workspace_size(),
        "rtol": 1e-4,
        "atol": 1e-5,
        "images": validate(False, runtime, naive=args.naive),
        "sequence_projection": validate(True, runtime, naive=args.naive),
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
