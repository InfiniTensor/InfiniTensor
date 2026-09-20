"""Hardware tests for criterion 3; no ORT execution inside the implementation."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from onnx import helper as h, TensorProto as T
import onnxruntime as ort
from pyinfinitensor import backend
from pyinfinitensor.onnx import DynamicOnnxStub


demo_dir = Path(__file__).resolve().parents[2] / "examples/dynamic_shape"
sys.path.insert(0, str(demo_dir))
spec = importlib.util.spec_from_file_location(
    "cuda_dynamic_demo", demo_dir / "cuda_demo.py"
)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)
pytestmark = pytest.mark.skipif(
    not hasattr(backend, "CudaRuntime"), reason="CUDA backend not built"
)


@pytest.fixture(scope="module")
def runtime():
    return backend.CudaRuntime(device=0, workspace_size=64 * 1024 * 1024)


@pytest.mark.parametrize("sequence", [False, True])
@pytest.mark.parametrize("naive", [False, True])
def test_cuda_dynamic_ort_and_cpu(runtime, sequence, naive):
    result = demo.validate(sequence, runtime, naive=naive)
    assert result["stats"]["runs"] == 5
    assert result["stats"]["graph_builds"] == 1


def test_workspace_configuration(runtime):
    assert runtime.workspace_size() == 64 * 1024 * 1024
    with pytest.raises(RuntimeError, match="workspace"):
        backend.CudaRuntime(workspace_size=0)
    with pytest.raises(RuntimeError, match="cache capacity"):
        backend.CudaRuntime(cuda_graph_cache_capacity=0)
    factory = backend.cuda_runtime(workspace_size=1024 * 1024)
    assert factory.workspace_size() == 1024 * 1024


def test_runtime_target_input_recovery_and_trim(runtime):
    model = h.make_model(
        h.make_graph(
            [h.make_node("Reshape", ["x", "target"], ["y"])],
            "runtime-target",
            [
                h.make_tensor_value_info("x", T.FLOAT, ["n", 4]),
                h.make_tensor_value_info("target", T.INT64, [2]),
            ],
            [h.make_tensor_value_info("y", T.FLOAT, [None, None])],
        ),
        opset_imports=[h.make_opsetid("", 13)],
        ir_version=9,
    )
    stub = DynamicOnnxStub(model, runtime)
    reference = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    for batch in [1, 2, 8, 3, 1]:
        x = np.arange(batch * 4, dtype=np.float32).reshape(batch, 4)
        feeds = {"x": x, "target": np.array([2, batch * 2], np.int64)}
        np.testing.assert_array_equal(
            stub.run(feeds)["y"], reference.run(None, feeds)[0]
        )
    with pytest.raises(ValueError, match="number of elements"):
        stub.run({"x": x, "target": np.array([3, 3], np.int64)})
    stub.handler.trim_memory()
    np.testing.assert_array_equal(stub.run(feeds)["y"], x.reshape(2, 2))


def test_invalid_fixed_dimension_does_not_launch(runtime):
    stub = DynamicOnnxStub(demo.model(), runtime)
    with pytest.raises(ValueError, match="fixed"):
        stub.run({"x": np.zeros((1, 4, 8, 8), np.float32)})
    assert stub.stats["runs"] == 0
    x = np.ones((1, 3, 8, 8), np.float32)
    np.testing.assert_array_equal(stub.run({"x": x})["y"], x.reshape(1, -1) + 0.25)
