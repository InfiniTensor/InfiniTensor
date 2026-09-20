"""Reproducible dynamic shape models and ONNX Runtime comparison."""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
import onnxruntime as ort
from pyinfinitensor import backend
from pyinfinitensor.onnx import DynamicOnnxStub


def model(sequence=False):
    init = [
        nh.from_array(np.array(0, np.int64), "index"),
        nh.from_array(np.array([0], np.int64), "axis"),
        nh.from_array(np.array([-1], np.int32), "minus32"),
    ]
    nodes = [
        h.make_node("Cast", ["minus32"], ["minus"], to=T.INT64),
        h.make_node("Shape", ["x"], ["s"]),
        h.make_node("Gather", ["s", "index"], ["b"], axis=0),
        h.make_node("Unsqueeze", ["b", "axis"], ["bv"]),
    ]
    if sequence:
        init += [
            nh.from_array(np.array([-1, 4], np.int64), "flat"),
            nh.from_array(np.arange(24, dtype=np.float32).reshape(4, 6) / 24, "weight"),
            nh.from_array(np.array([6], np.int64), "six"),
            nh.from_array(np.array(1, np.int64), "seq_index"),
        ]
        nodes += [
            h.make_node("Gather", ["s", "seq_index"], ["length"]),
            h.make_node("Unsqueeze", ["length", "axis"], ["lv"]),
            h.make_node("Concat", ["bv", "lv", "six"], ["target"], axis=0),
            h.make_node("Reshape", ["x", "flat"], ["flat_x"]),
            h.make_node("MatMul", ["flat_x", "weight"], ["projection"]),
            h.make_node("Relu", ["projection"], ["active"]),
            h.make_node("Reshape", ["active", "target"], ["y"]),
        ]
        input_shape, output_shape = ["batch", "sequence", 4], ["batch", "sequence", 6]
    else:
        init += [nh.from_array(np.array([0.25], np.float32), "bias")]
        nodes += [
            h.make_node("Concat", ["bv", "minus"], ["target"], axis=0),
            h.make_node("Reshape", ["x", "target"], ["flat_x"]),
            h.make_node("Add", ["flat_x", "bias"], ["biased"]),
            h.make_node("Relu", ["biased"], ["y"]),
        ]
        input_shape, output_shape = ["batch", 3, "height", None], ["batch", None]
    graph = h.make_graph(
        nodes,
        "dynamic_sequence" if sequence else "dynamic_images",
        [h.make_tensor_value_info("x", T.FLOAT, input_shape)],
        [h.make_tensor_value_info("y", T.FLOAT, output_shape)],
        init,
    )
    m = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)], ir_version=9)
    onnx.checker.check_model(m)
    return m


def compare(m, shapes, *, fold=True, naive=False):
    stub = DynamicOnnxStub(
        m, backend.cpu_runtime(), fold_constants=fold, use_naive_allocator=naive
    )
    ref = ort.InferenceSession(
        m.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(2026)
    rows, ids = [], None
    for shape in shapes:
        x = rng.standard_normal(shape).astype(np.float32)
        expected = ref.run(None, {"x": x})[0]
        start = time.perf_counter()
        actual = stub.run({"x": x})["y"]
        elapsed = (time.perf_counter() - start) * 1000
        np.testing.assert_equal(actual.shape, expected.shape)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        current = tuple(t.fuid() for t in stub.tensors.values())
        if ids is None:
            ids = current
        assert ids == current, "Backend tensors must persist across executions"
        rows.append(
            {
                "input": list(shape),
                "output": list(actual.shape),
                "max_abs_error": float(np.max(np.abs(actual - expected))),
                "latency_ms": elapsed,
            }
        )
    return {"runs": rows, "stats": stub.stats}, stub


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/dynamic_shape")
    args = parser.parse_args()
    path = Path(args.output)
    path.mkdir(parents=True, exist_ok=True)
    image_shapes = [
        (1, 3, 8, 8),
        (2, 3, 12, 10),
        (8, 3, 16, 12),
        (3, 3, 10, 8),
        (1, 3, 8, 8),
    ]
    sequence_shapes = [(1, 3, 4), (2, 8, 4), (8, 16, 4), (3, 5, 4), (1, 3, 4)]
    report = {}
    for name, seq, shapes in [
        ("images", False, image_shapes),
        ("sequence_projection", True, sequence_shapes),
    ]:
        m = model(seq)
        onnx.save(m, path / (name + ".onnx"))
        report[name], _ = compare(m, shapes)
    report["unfolded"], _ = compare(model(), image_shapes, fold=False)
    report["naive_allocator"], _ = compare(model(), image_shapes, naive=True)
    (path / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
