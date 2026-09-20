import importlib.util
from pathlib import Path
import numpy as np
import pytest
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper as nh
from pyinfinitensor import backend
from pyinfinitensor.onnx import DynamicOnnxStub
from pyinfinitensor.dynamic import _reshape, _evaluate

p = Path(__file__).resolve().parents[2] / "examples/dynamic_shape/demo.py"
spec = importlib.util.spec_from_file_location("dynamic_demo", p)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_chain_grow_shrink_and_weights():
    result, stub = demo.compare(
        demo.model(),
        [(1, 3, 4, 4), (2, 3, 5, 6), (8, 3, 8, 8), (3, 3, 5, 4), (1, 3, 4, 4)],
    )
    assert stub.input_specs["x"] == ("batch", 3, "height", None)
    assert result["stats"]["graph_builds"] == 1
    assert result["stats"]["folded_nodes"] == 1
    assert stub.tensors["bias"].copyout_numpy()[0] == 0.25


def test_sequence_projection():
    demo.compare(
        demo.model(True), [(1, 2, 4), (2, 5, 4), (8, 9, 4), (3, 3, 4), (1, 2, 4)]
    )


def test_folding_and_naive():
    shapes = [(1, 3, 4, 4), (3, 3, 5, 5), (1, 3, 4, 4)]
    demo.compare(demo.model(), shapes, fold=False)
    demo.compare(demo.model(), shapes, naive=True)


def test_runtime_target_input_and_recovery():
    m = h.make_model(
        h.make_graph(
            [h.make_node("Reshape", ["x", "target"], ["y"])],
            "target",
            [
                h.make_tensor_value_info("x", T.FLOAT, ["n", 4]),
                h.make_tensor_value_info("target", T.INT64, [2]),
            ],
            [h.make_tensor_value_info("y", T.FLOAT, [None, None])],
        ),
        opset_imports=[h.make_opsetid("", 13)],
        ir_version=9,
    )
    stub = DynamicOnnxStub(m, backend.cpu_runtime())
    for shape in [(2, 4), (4, 2), (1, 8), (8, 1), (2, 4)]:
        x = np.arange(8, dtype=np.float32).reshape(2, 4)
        y = stub.run({"x": x, "target": np.array(shape, np.int64)})["y"]
        np.testing.assert_equal(y, x.reshape(shape))
    with pytest.raises(ValueError, match="number of elements"):
        stub.run({"x": x, "target": np.array([3, 3], np.int64)})
    np.testing.assert_equal(
        stub.run({"x": x, "target": np.array([2, 4], np.int64)})["y"], x
    )


@pytest.mark.parametrize(
    "value, message",
    [
        (np.zeros((1, 4, 2, 2), np.float32), "fixed"),
        (np.zeros((1, 3, 2), np.float32), "rank"),
        (np.zeros((1, 3, 2, 2), np.float64), "dtype"),
        (np.zeros((0, 3, 2, 2), np.float32), "empty"),
    ],
)
def test_invalid_inputs(value, message):
    stub = DynamicOnnxStub(demo.model(), backend.cpu_runtime())
    with pytest.raises(ValueError, match=message):
        stub.run({"x": value})


@pytest.mark.parametrize("target", [[-1, -1], [3, -1], [-2, 4], [0, 0, 0], [3, 3]])
def test_invalid_reshape(target):
    with pytest.raises(ValueError):
        _reshape([2, 4], np.array(target, np.int64))


def test_shape_ops_dtypes_values_boundaries():
    shape = h.make_node("Shape", ["x"], ["s"], start=-2, end=3)
    np.testing.assert_equal(_evaluate(shape, {}, {"x": [2, 3, 4, 5]}), [4])
    gather = h.make_node("Gather", ["x", "i"], ["y"], axis=-1)
    v = {"x": np.array([2, 3, 4], np.int64), "i": np.array(-1, np.int64)}
    assert _evaluate(gather, v, {}).dtype == np.int64
    assert _evaluate(gather, v, {}) == 4
    v["i"] = np.array(3, np.int64)
    with pytest.raises(IndexError):
        _evaluate(gather, v, {})
    sq = h.make_node("Squeeze", ["x", "a"], ["y"])
    np.testing.assert_equal(
        _evaluate(
            sq, {"x": np.array([[2]], np.int64), "a": np.array([0], np.int64)}, {}
        ),
        [2],
    )
    un = h.make_node("Unsqueeze", ["x", "a"], ["y"])
    with pytest.raises(ValueError, match="duplicate"):
        _evaluate(un, {"x": np.array([2]), "a": np.array([0, 0])}, {})
    assert _reshape([2, 4], np.array([0, -1], np.int64)) == [2, 4]


def test_shared_symbols_and_missing_graph():
    m = demo.model()
    m.graph.input[0].type.tensor_type.shape.dim[2].dim_param = "batch"
    stub = DynamicOnnxStub(m, backend.cpu_runtime())
    with pytest.raises(ValueError, match="symbolic"):
        stub.run({"x": np.zeros((2, 3, 3, 4), np.float32)})
    m.graph.node[0].input[0] = "missing"
    with pytest.raises(ValueError, match="missing"):
        DynamicOnnxStub(m, backend.cpu_runtime())


def test_intermediate_shape_and_noncontiguous_input():
    m = demo.model()
    m.graph.node.extend(
        [
            h.make_node("Shape", ["y"], ["ys"]),
            h.make_node("Reshape", ["y", "ys"], ["z"]),
        ]
    )
    m.graph.output[0].name = "z"
    stub = DynamicOnnxStub(m, backend.cpu_runtime())
    x = np.arange(48, dtype=np.float32).reshape(1, 3, 4, 4)[:, :, :, ::-1]
    y = stub.run({"x": x})["z"]
    np.testing.assert_allclose(y, x.reshape(1, -1) + 0.25)


def test_capacity_reuse_and_trim():
    stub = DynamicOnnxStub(demo.model(), backend.cpu_runtime())
    for batch in [8, 4, 2, 1]:
        x = np.full((batch, 3, 4, 4), batch, np.float32)
        np.testing.assert_equal(stub.run({"x": x})["y"], x.reshape(batch, -1) + 0.25)
    stats = stub.handler.memory_stats()
    assert stats["activation_allocations"] == 1
    assert stats["activation_capacity_bytes"] > stats["activation_required_bytes"]
    stub.handler.trim_memory()
    trimmed = stub.handler.memory_stats()
    assert trimmed["activation_capacity_bytes"] == trimmed["activation_required_bytes"]
    assert trimmed["activation_allocations"] == 2
    np.testing.assert_equal(stub.run({"x": x})["y"], x.reshape(1, -1) + 0.25)


def test_shape_output_and_cast_squeeze_chain():
    m = demo.model()
    m.graph.node.extend(
        [
            h.make_node("Unsqueeze", ["target", "axis"], ["row"]),
            h.make_node("Squeeze", ["row", "axis"], ["vector"]),
            h.make_node("Cast", ["vector"], ["cast32"], to=T.INT32),
            h.make_node("Cast", ["cast32"], ["cast64"], to=T.INT64),
            h.make_node("Reshape", ["x", "cast64"], ["z"]),
        ]
    )
    m.graph.output[0].name = "z"
    m.graph.output.extend([h.make_tensor_value_info("cast64", T.INT64, [2])])
    stub = DynamicOnnxStub(m, backend.cpu_runtime())
    for b in [1, 4, 2]:
        x = np.arange(b * 12, dtype=np.float32).reshape(b, 3, 2, 2)
        result = stub.run({"x": x})
        np.testing.assert_equal(result["z"], x.reshape(b, -1))
        np.testing.assert_equal(result["cast64"], [b, -1])
        assert result["cast64"].dtype == np.int64
