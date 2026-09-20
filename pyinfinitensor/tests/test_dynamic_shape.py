"""Run with unittest discovery; ONNX Runtime is the independent oracle."""
import unittest
import numpy as np
import onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto, checker
from pyinfinitensor import backend
from pyinfinitensor.onnx import OnnxStub
from pyinfinitensor.shape_program import validate_shapes
from pyinfinitensor.dynamic_shape_demo import make_dynamic_model


class TestDynamicShape(unittest.TestCase):
    def test_repeated_shapes_ort_and_optimization(self):
        for spatial in (False, True):
            model = make_dynamic_model(spatial)
            session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
            for naive in (False, True):
                for optimize in (False, True):
                    with self.subTest(spatial=spatial, naive=naive, optimize=optimize):
                        stub = OnnxStub(model, backend.cpu_runtime(), naive,
                                        optimize_shape_program=optimize)
                        handler = stub.handler
                        self.assertEqual(stub.shape_program_stats["nodes_before"], 10)
                        self.assertEqual(stub.shape_program_stats["nodes_folded"], 6 if optimize else 0)
                        for batch, hw in zip((1, 2, 8, 3, 1), ((2, 3), (4, 5), (16, 17), (3, 2), (2, 3))):
                            dims = (batch, 2, *hw) if spatial else (batch, 2, 3)
                            x = np.random.default_rng(batch).normal(size=dims).astype(np.float32)
                            expected = session.run(None, {"x": x})
                            stub.set_input([dims])
                            stub.inputs["x"].copyin_numpy(x)
                            stub.run()
                            self.assertIs(stub.handler, handler)
                            for name, reference in zip(("y", "target"), expected):
                                actual = stub.outputs[name].copyout_numpy()
                                self.assertEqual(actual.shape, reference.shape)
                                self.assertEqual(actual.dtype, reference.dtype)
                                np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-5)
                            np.testing.assert_array_equal(stub.tensors["bias"].copyout_numpy(), [0.125])
                        if not naive:
                            stub.trim_memory()
                            stub.run()
                            np.testing.assert_allclose(stub.outputs["y"].copyout_numpy(), expected[0])

    def test_schema_validation_and_recovery(self):
        stub = OnnxStub(make_dynamic_model(), backend.cpu_runtime())
        for invalid in ([1, 4, 3], [1, 2], [0, 2, 3], [True, 2, 3], [1.5, 2, 3]):
            with self.assertRaises(ValueError):
                stub.set_input([invalid])
            self.assertEqual(stub.inputs["x"].shape(), [1, 2, 3])
        self.assertEqual(validate_shapes({"x": ("n", None, 3)}, [[2, 7, 3]]), {"x": [2, 7, 3]})
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_shapes({"x": ("n", "n")}, [[2, 3]])

    def test_shape_program_exceptions(self):
        model = make_dynamic_model()
        for init in model.graph.initializer:
            if init.name == "zero":
                init.CopyFrom(numpy_helper.from_array(np.asarray(100, dtype=np.int64), "zero"))
        with self.assertRaises((ValueError, IndexError)):
            OnnxStub(model, backend.cpu_runtime())
        model = make_dynamic_model()
        model.graph.node[0].input[0] = "reshaped"
        with self.assertRaisesRegex(ValueError, "Unable to resolve"):
            OnnxStub(model, backend.cpu_runtime())

    def test_export_preserves_dynamic_program(self):
        model = make_dynamic_model()
        stub = OnnxStub(model, backend.cpu_runtime())
        exported = stub.to_onnx("exported")
        checker.check_model(exported)
        self.assertEqual(len(exported.graph.node), len(model.graph.node))
        self.assertEqual(exported.graph.input[0].type.tensor_type.shape.dim[0].dim_param, "batch")

    def test_each_shape_operator_matches_ort(self):
        model = make_dynamic_model()
        # Include every intermediate as an observable output, including scalars.
        names = {v.name for v in model.graph.output}
        for node in model.graph.node[:10]:
            name = node.output[0]
            if name not in names:
                dtype = TensorProto.INT32 if name == "channels32" else TensorProto.INT64
                model.graph.output.append(helper.make_tensor_value_info(name, dtype, None))
        session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        stub = OnnxStub(model, backend.cpu_runtime())
        x = np.ones((3, 2, 3), np.float32)
        stub.set_input([x.shape])
        stub.inputs["x"].copyin_numpy(x)
        stub.run()
        for output, expected in zip(model.graph.output, session.run(None, {"x": x})):
            actual = stub.outputs[output.name].copyout_numpy()
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertEqual(actual.shape, expected.shape)
            np.testing.assert_array_equal(actual, expected)

    def test_invalid_runtime_reshape_rolls_back(self):
        model = make_dynamic_model()
        # [batch, batch, -1] is valid for batch=1,2,3 but not batch=8.
        model.graph.node[9].input[1] = "batch_vec"
        stub = OnnxStub(model, backend.cpu_runtime())
        stub.set_input([[2, 2, 3]])
        with self.assertRaisesRegex(ValueError, "Reshape"):
            stub.set_input([[8, 2, 3]])
        self.assertEqual(stub.inputs["x"].shape(), [2, 2, 3])
        x = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        stub.inputs["x"].copyin_numpy(x)
        stub.run()
        np.testing.assert_allclose(stub.outputs["y"].copyout_numpy(), x + 0.125)

    def test_shape_slice_and_negative_gather(self):
        model = make_dynamic_model()
        model.graph.node[0].attribute.extend([helper.make_attribute("start", -3),
                                             helper.make_attribute("end", 3)])
        for init in model.graph.initializer:
            if init.name == "zero":
                init.CopyFrom(numpy_helper.from_array(np.asarray(-3, np.int64), "zero"))
        stub = OnnxStub(model, backend.cpu_runtime())
        stub.set_input([[8, 2, 3]])
        np.testing.assert_array_equal(stub.tensors["target"].copyout_numpy(), [8, 2, -1])

    def test_unknown_dimensions_and_multiple_inputs(self):
        model = make_dynamic_model()
        model.graph.input[0].type.tensor_type.shape.dim[0].ClearField("dim_param")
        stub = OnnxStub(model, backend.cpu_runtime())
        self.assertIsNone(stub.input_schema["x"][0])
        stub.set_input([[5, 2, 3]])
        self.assertEqual(stub.outputs["y"].shape(), [5, 2, 3])
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_shapes({"x": ("n", 2), "z": ("n", 3)}, [[2, 2], [3, 3]])

    def test_initial_shapes_and_runtime_cast(self):
        model = make_dynamic_model()
        # Cast a partially known Shape before Gather: fixed components must
        # retain int32 dtype even when neighboring components are unknown.
        model.graph.node.insert(1, helper.make_node("Cast", ["dims"], ["dims32"], to=TensorProto.INT32))
        model.graph.node[4].input[0] = "dims32"
        model.graph.output.append(helper.make_tensor_value_info("channels", TensorProto.INT32, []))
        for optimized in (False, True):
            stub = OnnxStub(model, backend.cpu_runtime(), input_shapes=[[4, 2, 3]],
                            optimize_shape_program=optimized)
            self.assertEqual(stub.inputs["x"].shape(), [4, 2, 3])
            self.assertEqual(stub.outputs["channels"].copyout_numpy().dtype, np.int32)
            stub.set_input([[8, 2, 3]])
            np.testing.assert_array_equal(stub.tensors["dims32"].copyout_numpy(), [8, 2, 3])

    def test_shape_operator_invalid_axes_and_target_dtype(self):
        model = make_dynamic_model()
        model.graph.node[2].input[1] = "bad_axes"
        model.graph.initializer.append(numpy_helper.from_array(np.array([0, 0], np.int64), "bad_axes"))
        with self.assertRaises(ValueError):
            OnnxStub(model, backend.cpu_runtime())
        model = make_dynamic_model()
        model.graph.node.insert(10, helper.make_node("Cast", ["target"], ["bad_target"], to=TensorProto.INT32))
        model.graph.node[11].input[1] = "bad_target"
        with self.assertRaisesRegex(ValueError, "int64"):
            OnnxStub(model, backend.cpu_runtime())

    def test_dynamic_pool_capacity_is_reused(self):
        stub = OnnxStub(
            make_dynamic_model(), backend.cpu_runtime(),
            input_shapes=[[8, 2, 3]],
        )
        initial_capacity = stub.handler.activation_pool_capacity()
        initial_storage = stub.handler.activation_pool_storage_id()
        rows = []
        for batch in (8, 4, 2, 8):
            stub.set_input([[batch, 2, 3]])
            rows.append((
                stub.handler.planned_activation_bytes(),
                stub.handler.activation_pool_capacity(),
                stub.handler.activation_pool_storage_id(),
            ))
        self.assertTrue(all(planned <= capacity for planned, capacity, _ in rows))
        self.assertTrue(all(capacity == initial_capacity for _, capacity, _ in rows))
        self.assertTrue(all(storage == initial_storage for _, _, storage in rows))
        stub.trim_memory()
        self.assertLessEqual(stub.handler.activation_pool_capacity(), initial_capacity)

    def test_real_squeezenet_dynamic_hw(self):
        from pyinfinitensor.real_model_demo import validate

        result = validate()
        self.assertTrue(result["same_model_instance"])
        self.assertEqual(len(result["runs"]), 5)
        self.assertLessEqual(
            max(row["max_abs_error"] for row in result["runs"]), 2e-5
        )
        self.assertEqual(result["runs"][0]["input"], result["runs"][-1]["input"])

    @unittest.skipUnless(hasattr(backend, "cuda_runtime"), "CUDA backend not built")
    def test_cuda_dynamic_shape_and_graph(self):
        from pyinfinitensor.cuda_dynamic_shape_demo import validate

        result = validate()
        graph = result["cuda_graph"]
        self.assertEqual(
            graph["capture_count_after_first_cycle"],
            graph["capture_count_after_second_cycle"],
        )
        self.assertEqual(len(result["ordinary_cuda"]["runs"]), 5)
        self.assertEqual(len(graph["first_cycle"]), 5)


if __name__ == "__main__":
    unittest.main()
