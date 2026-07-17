import unittest
from unittest.mock import Mock, patch

import numpy as np
from onnx import TensorProto, checker, helper, numpy_helper

from pyinfinitensor import backend
from pyinfinitensor import onnx as onnx_frontend
from pyinfinitensor.onnx import OnnxStub, _parse_attribute


def make_model(nodes, inputs, outputs, initializers=(), check=True):
    graph = helper.make_graph(
        nodes,
        "onnxstub_test",
        inputs,
        outputs,
        initializer=list(initializers),
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    if check:
        checker.check_model(model)
    return model


def value_info(name, shape, dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, shape)


def initializer(name, values, dtype=None):
    array = np.asarray(values, dtype=dtype)
    return numpy_helper.from_array(array, name=name)


def import_model(model, runtime=None):
    runtime = backend.cpu_runtime() if runtime is None else runtime
    with patch.object(
        onnx_frontend, "simplify", side_effect=lambda candidate: (candidate, False)
    ):
        return OnnxStub(model, runtime)


def node_attribute(node, name):
    attribute = next(attr for attr in node.attribute if attr.name == name)
    return helper.get_attribute_value(attribute)


class TestOnnxStubImport(unittest.TestCase):
    def test_get_perf_time_returns_backend_value(self):
        stub = OnnxStub.__new__(OnnxStub)
        stub.handler = Mock()
        stub.handler.get_perf_time.return_value = 1.25

        self.assertEqual(stub.get_perf_time(), 1.25)

    def test_model_proto_is_not_modified(self):
        x = value_info("x", [2])
        y = value_info("y", [2])
        nodes = [
            helper.make_node("Identity", ["x"], ["middle"]),
            helper.make_node("Relu", ["middle"], ["y"]),
        ]
        model = make_model(nodes, [x], [y])
        before = model.SerializeToString()

        import_model(model)

        self.assertEqual(model.SerializeToString(), before)

    def test_unsorted_dag_creates_each_node_once(self):
        x = value_info("x", [2])
        y = value_info("y", [2])
        nodes = [
            helper.make_node("Relu", ["middle"], ["y"], name="relu"),
            helper.make_node("Identity", ["x"], ["middle"], name="identity"),
        ]
        model = make_model(nodes, [x], [y], check=False)

        stub = import_model(model)

        self.assertEqual(len(stub.handler.operators()), 2)

    def test_unresolved_graph_reports_nodes_and_inputs(self):
        y = value_info("y", [2])
        missing = make_model(
            [helper.make_node("Identity", ["missing"], ["y"], name="consumer")],
            [],
            [y],
            check=False,
        )
        with self.assertRaisesRegex(ValueError, "consumer.*missing"):
            import_model(missing)

        cycle = make_model(
            [
                helper.make_node("Identity", ["b"], ["a"], name="first"),
                helper.make_node("Identity", ["a"], ["b"], name="second"),
            ],
            [],
            [value_info("a", [2])],
            check=False,
        )
        with self.assertRaisesRegex(ValueError, "first.*b.*second.*a"):
            import_model(cycle)

    def test_attribute_parsing_does_not_share_state(self):
        with_alpha = helper.make_node("LeakyRelu", ["x"], ["y"], alpha=0.25)
        without_alpha = helper.make_node("Relu", ["x"], ["y"])

        self.assertEqual(_parse_attribute(with_alpha)["alpha"], 0.25)
        self.assertNotIn("alpha", _parse_attribute(without_alpha))

        defaults = {"axis": 1}
        parsed = _parse_attribute(with_alpha, defaults)
        parsed["axis"] = 2
        self.assertEqual(defaults, {"axis": 1})

    def test_constant_of_shape_is_explicitly_rejected(self):
        shape = value_info("shape", [2], TensorProto.INT64)
        output = value_info("output", [2, 2])
        model = make_model(
            [helper.make_node("ConstantOfShape", ["shape"], ["output"])],
            [shape],
            [output],
        )

        with self.assertRaisesRegex(NotImplementedError, "ConstantOfShape"):
            import_model(model)

    def test_set_input_requires_one_shape_per_input(self):
        model = make_model(
            [helper.make_node("Identity", ["x"], ["y"])],
            [value_info("x", [1, 2])],
            [value_info("y", [1, 2])],
        )
        stub = import_model(model)

        with self.assertRaisesRegex(ValueError, "expected 1, got 0"):
            stub.set_input([])
        with self.assertRaisesRegex(ValueError, "expected 1, got 2"):
            stub.set_input([[1, 2], [1, 2]])

    def test_initializer_is_restored_after_reallocation(self):
        weight = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        model = make_model(
            [helper.make_node("MatMul", ["x", "weight"], ["y"])],
            [value_info("x", [1, 2])],
            [value_info("y", [1, 2])],
            [initializer("weight", weight)],
        )
        stub = import_model(model)

        for batch in (2, 3):
            x = np.arange(batch * 2, dtype=np.float32).reshape(batch, 2)
            stub.set_input([[batch, 2]])
            stub.inputs["x"].copyin_numpy(x)
            stub.run()
            actual = np.asarray(stub.outputs["y"].copyout_float()).reshape(batch, 2)
            np.testing.assert_allclose(actual, x @ weight)


class TestStaticOnnxInputs(unittest.TestCase):
    def test_empty_optional_inputs_are_supported(self):
        cases = []

        max_value = initializer("max", np.array(1.0, dtype=np.float32))
        cases.append(
            make_model(
                [helper.make_node("Clip", ["x", "", "max"], ["y"])],
                [value_info("x", [2])],
                [value_info("y", [2])],
                [max_value],
            )
        )

        cases.append(
            make_model(
                [helper.make_node("Squeeze", ["x", ""], ["y"])],
                [value_info("x", [1, 2])],
                [value_info("y", [2])],
            )
        )

        cases.append(
            make_model(
                [helper.make_node("Split", ["x", ""], ["a", "b"], axis=0)],
                [value_info("x", [4])],
                [value_info("a", [2]), value_info("b", [2])],
            )
        )

        starts = initializer("starts", [0], np.int64)
        ends = initializer("ends", [2], np.int64)
        cases.append(
            make_model(
                [helper.make_node("Slice", ["x", "starts", "ends", "", ""], ["y"])],
                [value_info("x", [4])],
                [value_info("y", [2])],
                [starts, ends],
            )
        )

        pads = initializer("pads", [1, 1], np.int64)
        axes = initializer("axes", [1], np.int64)
        cases.append(
            make_model(
                [helper.make_node("Pad", ["x", "pads", "", "axes"], ["y"])],
                [value_info("x", [1, 2])],
                [value_info("y", [1, 4])],
                [pads, axes],
            )
        )

        cases.append(
            make_model(
                [helper.make_node("ReduceSum", ["x", ""], ["y"], keepdims=0)],
                [value_info("x", [2])],
                [value_info("y", [])],
            )
        )

        for model in cases:
            with self.subTest(op=model.graph.node[0].op_type):
                import_model(model)

    def test_required_static_input_rejects_runtime_tensor(self):
        model = make_model(
            [helper.make_node("Clip", ["x", "", "max"], ["y"])],
            [value_info("x", [2]), value_info("max", [], TensorProto.FLOAT)],
            [value_info("y", [2])],
        )

        with self.assertRaisesRegex(
            ValueError, r'Clip input 2 \("max"\) must be constant'
        ):
            import_model(model)

    def test_unsqueeze_without_axes_is_explicitly_rejected(self):
        model = make_model(
            [helper.make_node("Unsqueeze", ["x", ""], ["y"])],
            [value_info("x", [2])],
            [value_info("y", [1, 2])],
            check=False,
        )

        with self.assertRaisesRegex(ValueError, "Unsqueeze requires constant axes"):
            import_model(model)

    def test_pad_rejects_unsupported_mode_and_value(self):
        pads = initializer("pads", [1, 1], np.int64)
        reflect = make_model(
            [helper.make_node("Pad", ["x", "pads"], ["y"], mode="reflect")],
            [value_info("x", [2])],
            [value_info("y", [4])],
            [pads],
        )
        with self.assertRaisesRegex(NotImplementedError, "Pad mode"):
            import_model(reflect)

        value = initializer("value", np.array(1.0, dtype=np.float32))
        nonzero = make_model(
            [helper.make_node("Pad", ["x", "pads", "value"], ["y"])],
            [value_info("x", [2])],
            [value_info("y", [4])],
            [pads, value],
        )
        with self.assertRaisesRegex(NotImplementedError, "value of zero"):
            import_model(nonzero)

    def test_dropout_inference_and_unsupported_features(self):
        inference_training = initializer(
            "training", np.array(False, dtype=np.bool_)
        )
        inference_model = make_model(
            [helper.make_node("Dropout", ["x", "ratio", "training"], ["y"])],
            [
                value_info("x", [2]),
                value_info("ratio", [], TensorProto.FLOAT),
            ],
            [value_info("y", [2])],
            [inference_training],
        )
        import_model(inference_model)

        training = initializer("training", np.array(True, dtype=np.bool_))
        training_model = make_model(
            [helper.make_node("Dropout", ["x", "", "training"], ["y"])],
            [value_info("x", [2])],
            [value_info("y", [2])],
            [training],
        )
        with self.assertRaisesRegex(NotImplementedError, "training mode"):
            import_model(training_model)

        mask_model = make_model(
            [helper.make_node("Dropout", ["x"], ["y", "mask"])],
            [value_info("x", [2])],
            [value_info("y", [2]), value_info("mask", [2], TensorProto.BOOL)],
        )
        with self.assertRaisesRegex(NotImplementedError, "mask output"):
            import_model(mask_model)


class TestOnnxStubExport(unittest.TestCase):
    def assert_valid_export(self, model):
        checker.check_model(model)
        return model.graph.node[0]

    def test_conv_transpose_preserves_attributes(self):
        weight = np.ones((1, 1, 3, 3), dtype=np.float32)
        model = make_model(
            [
                helper.make_node(
                    "ConvTranspose",
                    ["x", "weight"],
                    ["y"],
                    pads=[1, 1, 1, 1],
                    strides=[2, 3],
                    dilations=[1, 2],
                    output_padding=[1, 2],
                )
            ],
            [value_info("x", [1, 1, 3, 3])],
            [value_info("y", [1, 1, 6, 11])],
            [initializer("weight", weight)],
        )
        node = self.assert_valid_export(import_model(model).to_onnx("export"))

        self.assertEqual(node_attribute(node, "pads"), [1, 1, 1, 1])
        self.assertEqual(node_attribute(node, "strides"), [2, 3])
        self.assertEqual(node_attribute(node, "dilations"), [1, 2])
        self.assertEqual(node_attribute(node, "output_padding"), [1, 2])

    def test_softmax_preserves_axis(self):
        model = make_model(
            [helper.make_node("Softmax", ["x"], ["y"], axis=1)],
            [value_info("x", [2, 3, 4])],
            [value_info("y", [2, 3, 4])],
        )
        node = self.assert_valid_export(import_model(model).to_onnx("export"))

        self.assertEqual(node_attribute(node, "axis"), 1)

    def test_split_exports_unequal_output_sizes(self):
        split = initializer("split", [1, 3], np.int64)
        model = make_model(
            [helper.make_node("Split", ["x", "split"], ["a", "b"], axis=1)],
            [value_info("x", [2, 4])],
            [value_info("a", [2, 1]), value_info("b", [2, 3])],
            [split],
        )
        exported = import_model(model).to_onnx("export")
        node = self.assert_valid_export(exported)
        exported_split = next(
            item for item in exported.graph.initializer if item.name == node.input[1]
        )

        self.assertEqual(numpy_helper.to_array(exported_split).tolist(), [1, 3])

    def test_expand_exports_shape_as_int64_input(self):
        shape = initializer("shape", [2, 3], np.int64)
        model = make_model(
            [helper.make_node("Expand", ["x", "shape"], ["y"])],
            [value_info("x", [1, 3])],
            [value_info("y", [2, 3])],
            [shape],
        )
        exported = import_model(model).to_onnx("export")
        node = self.assert_valid_export(exported)
        exported_shape = next(
            item for item in exported.graph.initializer if item.name == node.input[1]
        )

        self.assertEqual(len(node.input), 2)
        self.assertFalse(any(attr.name == "shape" for attr in node.attribute))
        self.assertEqual(exported_shape.data_type, TensorProto.INT64)
        self.assertEqual(numpy_helper.to_array(exported_shape).tolist(), [2, 3])

    def test_clip_exports_optional_inputs_and_bound_dtype(self):
        max_value = initializer("max", np.array(1.0, dtype=np.float16))
        model = make_model(
            [helper.make_node("Clip", ["x", "", "max"], ["y"])],
            [value_info("x", [2], TensorProto.FLOAT16)],
            [value_info("y", [2], TensorProto.FLOAT16)],
            [max_value],
        )
        exported = import_model(model).to_onnx("export")
        node = self.assert_valid_export(exported)
        exported_max = next(
            item for item in exported.graph.initializer if item.name == node.input[2]
        )

        self.assertEqual(node.input[1], "")
        self.assertEqual(exported_max.data_type, TensorProto.FLOAT16)
        self.assertEqual(numpy_helper.to_array(exported_max), np.float16(1.0))

    def test_lrn_exports_named_attributes(self):
        model = make_model(
            [
                helper.make_node(
                    "LRN",
                    ["x"],
                    ["y"],
                    alpha=0.001,
                    beta=0.5,
                    bias=2.0,
                    size=3,
                )
            ],
            [value_info("x", [1, 3, 2, 2])],
            [value_info("y", [1, 3, 2, 2])],
        )
        node = self.assert_valid_export(import_model(model).to_onnx("export"))

        self.assertAlmostEqual(node_attribute(node, "alpha"), 0.001)
        self.assertAlmostEqual(node_attribute(node, "beta"), 0.5)
        self.assertAlmostEqual(node_attribute(node, "bias"), 2.0)
        self.assertEqual(node_attribute(node, "size"), 3)

    def test_repeated_export_does_not_mutate_initializers(self):
        weight = initializer(
            "weight", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        )
        model = make_model(
            [helper.make_node("MatMul", ["x", "weight"], ["y"])],
            [value_info("x", [1, 2])],
            [value_info("y", [1, 2])],
            [weight],
        )
        stub = import_model(model)
        internal_before = {
            fuid: tensor.SerializeToString()
            for fuid, tensor in stub.initializer.items()
        }

        first = stub.to_onnx("export")
        second = stub.to_onnx("export")

        checker.check_model(first)
        checker.check_model(second)
        self.assertEqual(first.SerializeToString(), second.SerializeToString())
        self.assertEqual(
            {
                fuid: tensor.SerializeToString()
                for fuid, tensor in stub.initializer.items()
            },
            internal_before,
        )


class TestOnnxStubCuda(unittest.TestCase):
    @unittest.skipUnless(hasattr(backend, "cuda_runtime"), "CUDA backend not built")
    def test_dynamic_reallocation_restores_initializer(self):
        weight = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        model = make_model(
            [helper.make_node("MatMul", ["x", "weight"], ["y"])],
            [value_info("x", [1, 2])],
            [value_info("y", [1, 2])],
            [initializer("weight", weight)],
        )
        stub = import_model(model, backend.cuda_runtime())
        x = np.arange(6, dtype=np.float32).reshape(3, 2)

        stub.set_input([[3, 2]])
        stub.inputs["x"].copyin_numpy(x)
        stub.run()
        actual = np.asarray(stub.outputs["y"].copyout_float()).reshape(3, 2)

        np.testing.assert_allclose(actual, x @ weight, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
