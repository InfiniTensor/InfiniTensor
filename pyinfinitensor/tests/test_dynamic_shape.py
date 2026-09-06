"""CPU/ORT regression suite; CUDA cases require an explicit environment flag."""

import os
from pathlib import Path
import sys
import unittest

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh
import onnxruntime as ort
from pyinfinitensor import backend
from pyinfinitensor.onnx import OnnxStub

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/dynamic_shape"))
from models import batch_model


def session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


class TestDynamicShape(unittest.TestCase):
    def test_partial_evaluation_uses_fixed_dimension_not_placeholder(self):
        nodes = [
            h.make_node("Shape", ["x"], ["s"]),
            h.make_node("Gather", ["s", "axis"], ["y"], axis=0),
        ]
        model = h.make_model(
            h.make_graph(
                nodes,
                "partial",
                [h.make_tensor_value_info("x", T.FLOAT, ["batch", 3, None])],
                [h.make_tensor_value_info("y", T.INT64, [])],
                [nh.from_array(np.array(1, np.int64), "axis")],
            ),
            opset_imports=[h.make_opsetid("", 13)],
            ir_version=8,
        )
        stub = OnnxStub(model, backend.cpu_runtime())
        self.assertEqual(stub.shape_optimization["nodes_after"], 0)
        for shape in [(1, 3, 2), (4, 3, 7)]:
            self.assertEqual(stub.infer({"x": np.ones(shape, np.float32)})["y"], 3)

    def test_trained_cnn_dynamic_hw(self):
        artifacts = (
            Path(__file__).resolve().parents[2] / "examples/dynamic_shape/artifacts"
        )
        model = onnx.load(artifacts / "digits_cnn.onnx")
        samples = np.load(artifacts / "digits_samples.npz")
        stub = OnnxStub(model, backend.cpu_runtime())
        reference = session(model)
        for batch, height, width in [
            (1, 8, 8),
            (2, 12, 10),
            (4, 16, 16),
            (3, 10, 12),
            (1, 8, 8),
        ]:
            x = samples["images"][:batch, :, (np.arange(height) * 8 // height), :]
            x = np.ascontiguousarray(x[:, :, :, (np.arange(width) * 8 // width)])
            expected = reference.run(None, {"images": x})[0]
            actual = stub.infer({"images": x})["logits"]
            self.assertEqual(actual.shape, (batch, 10))
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
            if height == width == 8:
                np.testing.assert_allclose(
                    actual, samples["logits"][:batch], rtol=1e-4, atol=1e-5
                )

    def test_failed_dynamic_reshape_restores_instance(self):
        model = batch_model()
        # Use a dynamic third dimension; the target still requires six features.
        model.graph.input[0].type.tensor_type.shape.dim[2].ClearField("dim_value")
        model.graph.input[0].type.tensor_type.shape.dim[2].dim_param = "width"
        # A [batch, 2, width] placeholder cannot satisfy the original target,
        # so use a target whose fixed feature count is two for this fixture.
        replacement = batch_model(2)
        model.graph.initializer[0].CopyFrom(replacement.graph.initializer[0])
        stub = OnnxStub(model, backend.cpu_runtime())
        with self.assertRaisesRegex(RuntimeError, "element count mismatch"):
            stub.infer({"x": np.ones((2, 2, 3), np.float32)})
        x = np.ones((2, 2, 1), np.float32)
        np.testing.assert_allclose(
            stub.infer({"x": x})["y"], session(model).run(None, {"x": x})[0]
        )

    def test_five_shapes_and_partial_evaluation(self):
        model = batch_model()
        reference = session(model)
        rng = np.random.default_rng(7)
        for optimized, naive in [(False, False), (True, False), (True, True)]:
            stub = OnnxStub(
                model,
                backend.cpu_runtime(),
                optimize_shapes=optimized,
                use_naive_allocator=naive,
            )
            self.assertEqual(stub.shape_optimization["nodes_before"], 7)
            self.assertEqual(
                stub.shape_optimization["nodes_after"], 4 if optimized else 7
            )
            for batch in [1, 2, 8, 3, 1]:
                x = rng.normal(size=(batch, 2, 3)).astype(np.float32)
                actual = stub.infer({"x": x})["y"]
                expected = reference.run(None, {"x": x})[0]
                self.assertEqual(actual.shape, expected.shape)
                np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
            exported = stub.to_onnx("round_trip")
            self.assertEqual(
                exported.graph.input[0].type.tensor_type.shape.dim[0].dim_param, "batch"
            )
            self.assertTrue(any(n.op_type == "Shape" for n in exported.graph.node))

    def test_fixed_symbolic_unknown_and_input_validation(self):
        model = h.make_model(
            h.make_graph(
                [h.make_node("Add", ["x", "z"], ["y"])],
                "contracts",
                [
                    h.make_tensor_value_info("x", T.FLOAT, ["b", 3, None]),
                    h.make_tensor_value_info("z", T.FLOAT, ["b", 3, None]),
                ],
                [h.make_tensor_value_info("y", T.FLOAT, ["b", 3, None])],
            ),
            opset_imports=[h.make_opsetid("", 13)],
            ir_version=8,
        )
        stub = OnnxStub(model, backend.cpu_runtime())
        self.assertEqual(
            [dim.kind for dim in stub.input_specs["x"]],
            ["symbolic", "fixed", "unknown"],
        )
        for shapes, message in [
            ([[2, 4, 2], [2, 3, 2]], "fixed"),
            ([[2, 3, 2], [3, 3, 2]], "symbol b"),
            ([[2, 3], [2, 3]], "rank"),
            ([[0, 3, 2], [0, 3, 2]], "positive"),
            ([[1.5, 3, 2], [1, 3, 2]], "integer"),
        ]:
            with self.assertRaisesRegex(ValueError, message):
                stub.set_input(shapes)
            self.assertEqual(stub.getShape("x"), [1, 3, 1])
        x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)[:, :, ::-1]
        np.testing.assert_equal(stub.infer({"x": x, "z": x})["y"], x * 2)
        with self.assertRaisesRegex(ValueError, "dtype"):
            stub.infer({"x": x.astype(np.float64), "z": x})

    def test_shape_output_cast_squeeze_and_scalar_dtype(self):
        nodes = [
            h.make_node("Shape", ["x"], ["s"]),
            h.make_node("Gather", ["s", "index"], ["g"], axis=0),
            h.make_node("Cast", ["g"], ["small"], to=T.INT32),
            h.make_node("Cast", ["small"], ["wide"], to=T.INT64),
            h.make_node("Unsqueeze", ["wide", "axes"], ["v"]),
            h.make_node("Squeeze", ["v", "axes"], ["scalar"]),
        ]
        model = h.make_model(
            h.make_graph(
                nodes,
                "shape_outputs",
                [h.make_tensor_value_info("x", T.FLOAT, ["b", 2, "w"])],
                [
                    h.make_tensor_value_info("scalar", T.INT64, []),
                    h.make_tensor_value_info("s", T.INT64, [3]),
                ],
                [
                    nh.from_array(np.array(-1, np.int64), "index"),
                    nh.from_array(np.array([0], np.int64), "axes"),
                ],
            ),
            opset_imports=[h.make_opsetid("", 13)],
            ir_version=8,
        )
        stub = OnnxStub(model, backend.cpu_runtime(), optimize_shapes=False)
        for width in [1, 7, 2]:
            result = stub.infer({"x": np.ones((3, 2, width), np.float32)})
            self.assertEqual(result["scalar"].dtype, np.int64)
            self.assertEqual(result["scalar"].shape, ())
            self.assertEqual(result["scalar"].item(), width)
            np.testing.assert_equal(result["s"], [3, 2, width])

    def test_capacity_reuse_reduces_real_allocations(self):
        counts = []
        for reuse in [False, True]:
            stub = OnnxStub(batch_model(), backend.cpu_runtime(), memory_reuse=reuse)
            for batch in [1, 2, 8, 3, 1] * 3:
                stub.infer({"x": np.ones((batch, 2, 3), np.float32)})
            counts.append(stub.memory_stats()["activation_allocations"])
        self.assertLess(counts[1], counts[0])


@unittest.skipUnless(
    os.environ.get("INFINITENSOR_TEST_CUDA") == "1",
    "set INFINITENSOR_TEST_CUDA=1 with a CUDA build and GPU",
)
class TestDynamicShapeCuda(unittest.TestCase):
    def test_cpu_cuda_and_graph_cache(self):
        model = batch_model(64)
        runtime = backend.cuda_runtime()
        cpu = OnnxStub(model, backend.cpu_runtime())
        gpu = OnnxStub(model, runtime)
        reference = session(model)
        rng = np.random.default_rng(8)
        for batch in [1, 2, 8, 3, 1]:
            x = rng.normal(size=(batch, 2, 32)).astype(np.float32)
            expected = reference.run(None, {"x": x})[0]
            for result in [
                cpu.infer({"x": x})["y"],
                gpu.infer({"x": x})["y"],
                gpu.infer({"x": x}, cuda_graph=True)["y"],
            ]:
                np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)
        # The pool has reached its high-water capacity. Revisited shapes with
        # identical addresses must reuse their captured execution state.
        for batch in [8, 3, 1]:
            gpu.infer({"x": np.ones((batch, 2, 32), np.float32)}, cuda_graph=True)
        captures = runtime.cuda_graph_capture_count()
        for batch in [8, 3, 1] * 2:
            gpu.infer({"x": np.ones((batch, 2, 32), np.float32)}, cuda_graph=True)
        self.assertEqual(runtime.cuda_graph_capture_count(), captures)
        gpu.trim_memory()
        x = np.ones((1, 2, 32), np.float32)
        np.testing.assert_allclose(
            gpu.infer({"x": x}, cuda_graph=True)["y"],
            reference.run(None, {"x": x})[0],
            rtol=1e-4,
            atol=1e-5,
        )
        self.assertGreater(runtime.cuda_graph_capture_count(), captures)


if __name__ == "__main__":
    unittest.main()
