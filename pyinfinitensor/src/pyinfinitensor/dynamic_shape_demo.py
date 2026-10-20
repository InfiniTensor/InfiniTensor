"""Reproducible CPU dynamic-shape demo: python -m pyinfinitensor.dynamic_shape_demo."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto
from pyinfinitensor import backend
from pyinfinitensor.onnx import OnnxStub


def make_dynamic_model(spatial=False):
    vi = helper.make_tensor_value_info
    node = helper.make_node

    def init(name, value):
        return numpy_helper.from_array(np.asarray(value, dtype=np.int64), name)

    dims = ["batch", 2, "height", "width"] if spatial else ["batch", 2, 3]
    nodes = [
        node("Shape", ["x"], ["dims"]),
        node("Gather", ["dims", "zero"], ["batch"]),
        node("Unsqueeze", ["batch", "axes"], ["batch_vec"]),
        node("Gather", ["dims", "one"], ["channels"]),
        node("Unsqueeze", ["channels", "axes"], ["channels_vec"]),
        node("Cast", ["channels_vec"], ["channels32"], to=TensorProto.INT32),
        node("Cast", ["channels32"], ["channels64"], to=TensorProto.INT64),
        node("Squeeze", ["channels64", "axes"], ["channels_scalar"]),
        node("Unsqueeze", ["channels_scalar", "axes"], ["fixed"]),
        node("Concat", ["batch_vec", "fixed", "infer"], ["target"], axis=0),
        node("Reshape", ["x", "target"], ["reshaped"]),
        node("Add", ["reshaped", "bias"], ["y"]),
    ]
    constants = [init("zero", 0), init("one", 1), init("axes", [0]), init("infer", [-1]),
                 numpy_helper.from_array(np.asarray([0.125], dtype=np.float32), "bias")]
    graph = helper.make_graph(nodes, "dynamic_shape", [vi("x", TensorProto.FLOAT, dims)],
        [vi("y", TensorProto.FLOAT, ["batch", 2, "spatial"]),
         vi("target", TensorProto.INT64, [3])], constants)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    # Only IR8 features are used; do not inherit the installed ONNX default IR.
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def benchmark(repeats=100):
    result = {"environment": {"python": platform.python_version(),
              "numpy": np.__version__, "onnx": onnx.__version__,
              "onnxruntime": ort.__version__, "platform": platform.platform()},
              "tolerance": {"rtol": 1e-4, "atol": 1e-5}, "cases": []}
    for spatial in (False, True):
        model = make_dynamic_model(spatial)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        reference = ort.InferenceSession(model.SerializeToString(), options,
                                        providers=["CPUExecutionProvider"])
        for optimized in (False, True):
            stub = OnnxStub(model, backend.cpu_runtime(), optimize_shape_program=optimized)
            handler = stub.handler
            rows = []
            for batch, hw in zip((1, 2, 8, 3, 1), ((2, 3), (4, 5), (16, 17), (3, 2), (2, 3))):
                dims = (batch, 2, *hw) if spatial else (batch, 2, 3)
                x = np.random.default_rng(batch).normal(size=dims).astype(np.float32)
                expected, target = reference.run(None, {"x": x})
                stub.set_input([dims])
                stub.inputs["x"].copyin_numpy(x)
                stub.run()
                actual = stub.outputs["y"].copyout_numpy()
                assert stub.handler is handler
                assert actual.shape == expected.shape
                np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
                np.testing.assert_array_equal(stub.outputs["target"].copyout_numpy(), target)
                np.testing.assert_array_equal(stub.tensors["bias"].copyout_numpy(), [0.125])
                rows.append({"input": list(dims), "output": list(actual.shape),
                             "target": target.tolist(),
                             "max_abs_error": float(np.max(np.abs(actual - expected)))})
            # Warm-up then time shape evaluation separately. This is not a
            # claim about whole-model speedup or native allocator performance.
            concrete = {"x": list(dims)}
            for _ in range(10):
                stub._shape_program.evaluate(concrete)
            timings = []
            for _ in range(repeats):
                start = time.perf_counter_ns()
                stub._shape_program.evaluate(concrete)
                timings.append((time.perf_counter_ns() - start) / 1000)
            result["cases"].append({"model": "synthetic_hw" if spatial else "synthetic_batch",
                "optimized": optimized, "same_model_instance": True,
                "shape_program": stub.shape_program_stats, "runs": rows,
                "shape_eval_median_us": float(np.median(timings)), "repeats": repeats})
    result["memory_benchmark"] = benchmark_memory_reuse(repeats)
    return result


def benchmark_memory_reuse(repeats=100):
    """Compare exact-fit trimming with the default high-watermark pool."""
    model = make_dynamic_model()
    sequence = (1, 2, 8, 3, 1)

    def run_strategy(trim_each_shape):
        stub = OnnxStub(model, backend.cpu_runtime(), input_shapes=[[8, 2, 3]])
        previous_storage = stub.handler.activation_pool_storage_id()
        reallocations = 0
        peak_capacity = stub.handler.activation_pool_capacity()
        timings = []
        for cycle in range(repeats):
            for offset, batch in enumerate(sequence):
                dims = (batch, 2, 3)
                data = np.full(dims, cycle + offset / 10, dtype=np.float32)
                started = time.perf_counter_ns()
                stub.set_input([dims])
                current = stub.handler.activation_pool_storage_id()
                reallocations += current != previous_storage
                previous_storage = current
                if trim_each_shape:
                    stub.trim_memory()
                    current = stub.handler.activation_pool_storage_id()
                    reallocations += current != previous_storage
                    previous_storage = current
                peak_capacity = max(peak_capacity, stub.handler.activation_pool_capacity())
                stub.inputs["x"].copyin_numpy(data)
                stub.run()
                actual = stub.outputs["y"].copyout_numpy()
                timings.append((time.perf_counter_ns() - started) / 1000)
                np.testing.assert_allclose(actual, data + 0.125, rtol=1e-4, atol=1e-5)
        return {
            "storage_reallocations": int(reallocations),
            "median_end_to_end_us": float(np.median(timings)),
            "peak_capacity_bytes": int(peak_capacity),
            "measurements": len(timings),
        }

    return {
        "shape_sequence": [[batch, 2, 3] for batch in sequence],
        "cycles": repeats,
        "exact_fit": run_strategy(True),
        "high_water_reuse": run_strategy(False),
        "timing_scope": "set_input + optional trim + copyin + run + copyout",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/dynamic-shape")
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for spatial in (False, True):
        onnx.save(make_dynamic_model(spatial), output / ("hw.onnx" if spatial else "batch.onnx"))
    result = benchmark(args.repeats)
    text = json.dumps(result, indent=2)
    (output / "results.json").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
