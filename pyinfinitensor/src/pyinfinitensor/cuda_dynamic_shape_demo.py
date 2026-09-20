"""CUDA and CUDA Graph validation for the dynamic Shape subgraph."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from pyinfinitensor import backend
from pyinfinitensor.dynamic_shape_demo import make_dynamic_model
from pyinfinitensor.onnx import OnnxStub


SHAPES = [(1, 2, 3), (2, 2, 3), (8, 2, 3), (3, 2, 3), (1, 2, 3)]


def _run_sequence(stub, reference, use_graph):
    rows = []
    for index, dims in enumerate(SHAPES):
        data = np.random.default_rng(700 + index).normal(size=dims).astype(np.float32)
        expected, target = reference.run(None, {"x": data})
        started = time.perf_counter_ns()
        stub.set_input([dims])
        stub.inputs["x"].copyin_numpy(data)
        if use_graph:
            stub.run_with_cudagraph()
        else:
            stub.run()
        actual = stub.outputs["y"].copyout_numpy()
        elapsed_us = (time.perf_counter_ns() - started) / 1000
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        np.testing.assert_array_equal(stub.outputs["target"].copyout_numpy(), target)
        rows.append({
            "input": list(dims), "output": list(actual.shape),
            "max_abs_error": float(np.max(np.abs(actual - expected))),
            "end_to_end_us": elapsed_us,
            "planned_activation_bytes": stub.handler.planned_activation_bytes(),
            "activation_pool_capacity": stub.handler.activation_pool_capacity(),
            "activation_pool_storage_id": stub.handler.activation_pool_storage_id(),
        })
    return rows


def validate():
    if not hasattr(backend, "cuda_runtime"):
        raise RuntimeError("This backend was built without CUDA")
    model = make_dynamic_model()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    reference = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )

    normal_runtime = backend.cuda_runtime()
    normal = OnnxStub(model, normal_runtime, input_shapes=[[8, 2, 3]])
    normal_handler = normal.handler
    normal_rows = _run_sequence(normal, reference, False)
    assert normal.handler is normal_handler

    graph_runtime = backend.CudaRuntime(device=0, cuda_graph_cache_capacity=16)
    graph = OnnxStub(model, graph_runtime, input_shapes=[[8, 2, 3]])
    graph_handler = graph.handler
    first_rows = _run_sequence(graph, reference, True)
    first_capture_count = graph_runtime.cuda_graph_capture_count()
    first_cache_size = graph_runtime.cuda_graph_cache_size()
    second_rows = _run_sequence(graph, reference, True)
    second_capture_count = graph_runtime.cuda_graph_capture_count()
    second_cache_size = graph_runtime.cuda_graph_cache_size()
    assert graph.handler is graph_handler
    if second_capture_count != first_capture_count:
        raise AssertionError(
            "Previously seen shapes were captured again: {} -> {}".format(
                first_capture_count, second_capture_count
            )
        )

    storage_before_growth = graph.handler.activation_pool_storage_id()

    def run_after_growth(dims, seed):
        data = np.random.default_rng(seed).normal(size=dims).astype(np.float32)
        expected, target = reference.run(None, {"x": data})
        graph.set_input([dims])
        graph.inputs["x"].copyin_numpy(data)
        graph.run_with_cudagraph()
        actual = graph.outputs["y"].copyout_numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        np.testing.assert_array_equal(graph.outputs["target"].copyout_numpy(), target)
        return {
            "input": list(dims), "output": list(actual.shape),
            "max_abs_error": float(np.max(np.abs(actual - expected))),
            "storage_id": graph.handler.activation_pool_storage_id(),
            "capacity": graph.handler.activation_pool_capacity(),
            "capture_count": graph_runtime.cuda_graph_capture_count(),
            "cache_size": graph_runtime.cuda_graph_cache_size(),
        }

    grown = run_after_growth((32, 2, 3), 900)
    if grown["storage_id"] == storage_before_growth:
        raise AssertionError("The growth case did not replace activation storage")
    returned = run_after_growth((1, 2, 3), 901)
    if grown["capture_count"] <= second_capture_count:
        raise AssertionError("The grown shape did not trigger CUDA Graph capture")
    if returned["capture_count"] <= grown["capture_count"]:
        raise AssertionError("The old-address CUDA Graph was reused after storage changed")

    storage_ids = [row["activation_pool_storage_id"] for row in first_rows]
    reallocations = sum(a != b for a, b in zip(storage_ids, storage_ids[1:]))
    capacity = [row["activation_pool_capacity"] for row in first_rows]
    planned = [row["planned_activation_bytes"] for row in first_rows]
    return {
        "device": "CUDA",
        "shape_sequence": [list(value) for value in SHAPES],
        "ordinary_cuda": {"same_model_instance": True, "runs": normal_rows},
        "cuda_graph": {
            "same_model_instance": True,
            "first_cycle": first_rows,
            "second_cycle": second_rows,
            "capture_count_after_first_cycle": first_capture_count,
            "capture_count_after_second_cycle": second_capture_count,
            "cache_size_after_first_cycle": first_cache_size,
            "cache_size_after_second_cycle": second_cache_size,
            "storage_invalidation": {
                "storage_id_before_growth": storage_before_growth,
                "grown_shape": grown,
                "return_to_previous_shape": returned,
            },
        },
        "memory_reuse": {
            "storage_ids": storage_ids,
            "storage_reallocations_during_first_cycle": reallocations,
            "planned_activation_bytes": planned,
            "activation_pool_capacity": capacity,
            "peak_capacity_bytes": max(capacity),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/cuda-dynamic-shape")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result = validate()
    text = json.dumps(result, indent=2)
    (output / "results.json").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
