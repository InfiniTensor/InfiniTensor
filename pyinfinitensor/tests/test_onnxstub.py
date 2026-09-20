import unittest
from unittest.mock import Mock, patch

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None
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


def import_model(model, runtime=None, use_naive_allocator=False):
    runtime = backend.cpu_runtime() if runtime is None else runtime
    with patch.object(
        onnx_frontend, "simplify", side_effect=lambda candidate: (candidate, False)
    ):
        return OnnxStub(model, runtime, use_naive_allocator=use_naive_allocator)


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

    def test_constant_of_shape_dynamic_input_is_supported(self):
        shape = value_info("shape", [2], TensorProto.INT64)
        output = value_info("output", [2, 2])
        model = make_model(
            [helper.make_node("ConstantOfShape", ["shape"], ["output"])],
            [shape],
            [output],
        )

        stub = import_model(model)
        stub.inputs["shape"].copyin_numpy(np.array([2, 2], dtype=np.int64))
        stub.set_input([[2]])
        stub.run()
        actual = np.asarray(stub.outputs["output"].copyout_float()).reshape(2, 2)
        np.testing.assert_array_equal(actual, np.zeros((2, 2), dtype=np.float32))

    def test_constant_of_shape_static_is_folded(self):
        shape = initializer("shape", np.array([2, 3], dtype=np.int64))
        output = value_info("output", [2, 3])
        model = make_model(
            [
                helper.make_node(
                    "ConstantOfShape", ["shape"], ["output"],
                    value=numpy_helper.from_array(
                        np.array([2.5], dtype=np.float32), name="value"
                    ),
                )
            ],
            [],
            [output],
            [shape],
        )
        stub = import_model(model)
        stub.run()
        actual = np.asarray(stub.outputs["output"].copyout_float()).reshape(2, 3)
        np.testing.assert_array_equal(actual, np.full((2, 3), 2.5, dtype=np.float32))

    def test_constant_of_shape_dynamic_input_matches_ort(self):
        shape = value_info("shape", [2], TensorProto.INT64)
        output = value_info("output", [None, None], TensorProto.FLOAT)
        value = numpy_helper.from_array(
            np.array([1.25], dtype=np.float32), name="fill_value"
        )
        model = make_model(
            [helper.make_node("ConstantOfShape", ["shape"], ["output"], value=value)],
            [shape],
            [output],
        )
        stub = import_model(model)
        ort_session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        ) if ort is not None else None
        for dims in ([2, 3], [3, 1], [1, 4]):
            shape_values = np.asarray(dims, dtype=np.int64)
            stub.inputs["shape"].copyin_numpy(shape_values)
            stub.set_input([[2]])
            stub.run()
            actual = np.asarray(stub.outputs["output"].copyout_float()).reshape(dims)
            np.testing.assert_array_equal(
                actual, np.full(dims, 1.25, dtype=np.float32)
            )
            if ort_session is not None:
                expected = ort_session.run(["output"], {"shape": shape_values})[0]
                np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

    def test_constant_of_shape_from_shape_subgraph_matches_ort(self):
        x = value_info("x", [None, 2, 3])
        output = value_info("output", [None, None, None])
        value = numpy_helper.from_array(
            np.array([3.0], dtype=np.float32), name="fill_value"
        )
        model = make_model(
            [
                helper.make_node("Shape", ["x"], ["runtime_shape"]),
                helper.make_node(
                    "ConstantOfShape", ["runtime_shape"], ["output"], value=value
                ),
            ],
            [x],
            [output],
        )
        stub = import_model(model)
        ort_session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        ) if ort is not None else None
        for batch in (1, 5, 2, 1):
            values = np.zeros((batch, 2, 3), dtype=np.float32)
            stub.set_input([[batch, 2, 3]])
            stub.inputs["x"].copyin_numpy(values)
            stub.run()
            actual = np.asarray(stub.outputs["output"].copyout_float()).reshape(
                batch, 2, 3
            )
            np.testing.assert_array_equal(
                actual, np.full((batch, 2, 3), 3.0, dtype=np.float32)
            )
            if ort_session is not None:
                expected = ort_session.run(["output"], {"x": values})[0]
                np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(hasattr(backend, "cuda_runtime"), "CUDA backend not built")
    def test_constant_of_shape_dynamic_cuda(self):
        shape = value_info("shape", [2], TensorProto.INT64)
        output = value_info("output", [None, None], TensorProto.FLOAT)
        value = numpy_helper.from_array(
            np.array([2.0], dtype=np.float32), name="fill_value"
        )
        model = make_model(
            [helper.make_node("ConstantOfShape", ["shape"], ["output"], value=value)],
            [shape],
            [output],
        )
        stub = import_model(
            model, backend.cuda_runtime(workspace_size=64 << 20)
        )
        for dims in ([2, 2], [4, 1], [1, 5]):
            shape_values = np.asarray(dims, dtype=np.int64)
            stub.inputs["shape"].copyin_numpy(shape_values)
            stub.set_input([[2]])
            stub.run()
            actual = np.asarray(stub.outputs["output"].copyout_float()).reshape(dims)
            np.testing.assert_array_equal(actual, np.full(dims, 2.0, dtype=np.float32))

    def test_reshape_allowzero_semantics(self):
        shape = initializer("shape", np.array([0, 3], dtype=np.int64))
        model = make_model(
            [helper.make_node("Reshape", ["x", "shape"], ["y"], allowzero=1)],
            [value_info("x", [0, 3])],
            [value_info("y", [0, 3])],
            [shape],
        )
        stub = import_model(model)
        stub.run()
        self.assertEqual(stub.getShape("y"), [0, 3])

        invalid = make_model(
            [helper.make_node("Reshape", ["x", "shape"], ["y"], allowzero=1)],
            [value_info("x", [2, 3])],
            [value_info("y", [2, 3])],
            [initializer("shape", np.array([0, -1], dtype=np.int64))],
            check=False,
        )
        with self.assertRaisesRegex(RuntimeError, "allowzero"):
            import_model(invalid)

    def test_set_input_requires_one_shape_per_input(self):
        model = make_model(
            [helper.make_node("Identity", ["x"], ["y"])],
            [value_info("x", [None, 2])],
            [value_info("y", [None, 2])],
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
            [value_info("x", [None, 2])],
            [value_info("y", [None, 2])],
            [initializer("weight", weight)],
        )
        for use_naive_allocator in (False, True):
            with self.subTest(use_naive_allocator=use_naive_allocator):
                stub = import_model(model, use_naive_allocator=use_naive_allocator)
                for batch in (2, 1, 8192, 3):
                    x = np.arange(batch * 2, dtype=np.float32).reshape(batch, 2)
                    stub.set_input([[batch, 2]])
                    stub.inputs["x"].copyin_numpy(x)
                    stub.run()
                    actual = np.asarray(stub.outputs["y"].copyout_float()).reshape(
                        batch, 2
                    )
                    np.testing.assert_allclose(actual, x @ weight)
                if not use_naive_allocator:
                    stub.trim_memory()
                    stub.run()
                    actual = np.asarray(stub.outputs["y"].copyout_float()).reshape(
                        3, 2
                    )
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
            [value_info("x", [None, 2])],
            [value_info("y", [None, 2])],
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
            [value_info("x", [None, 2])],
            [value_info("y", [None, 2])],
            [initializer("weight", weight)],
        )
        for use_naive_allocator in (False, True):
            with self.subTest(use_naive_allocator=use_naive_allocator):
                stub = import_model(
                    model,
                    backend.cuda_runtime(workspace_size=64 << 20),
                    use_naive_allocator=use_naive_allocator,
                )
                for batch in (3, 1, 8192, 2):
                    x = np.arange(batch * 2, dtype=np.float32).reshape(batch, 2)
                    stub.set_input([[batch, 2]])
                    stub.inputs["x"].copyin_numpy(x)
                    stub.run()
                    actual = np.asarray(stub.outputs["y"].copyout_float()).reshape(
                        batch, 2
                    )
                    np.testing.assert_allclose(actual, x @ weight, rtol=1e-5, atol=1e-6)
                if not use_naive_allocator:
                    stub.trim_memory()
                    stub.run()
                    actual = np.asarray(stub.outputs["y"].copyout_float()).reshape(
                        2, 2
                    )
                    np.testing.assert_allclose(actual, x @ weight, rtol=1e-5, atol=1e-6)


class TestDynamicShapeSubgraph(unittest.TestCase):
    def make_model(self):
        # X[N, 2, 3] -> Shape -> Gather(batch) -> Unsqueeze ->
        # Concat([batch], [6]) -> Reshape(X, [N, 6]).
        x = value_info("x", [None, 2, 3])
        y = value_info("y", [None, 6])
        indices = initializer("batch_index", np.array(0, dtype=np.int64))
        tail = initializer("tail", np.array([6], dtype=np.int64))
        nodes = [
            helper.make_node("Shape", ["x"], ["x_shape"], name="shape"),
            helper.make_node(
                "Gather", ["x_shape", "batch_index"], ["batch_value"],
                axis=0, name="gather_batch"
            ),
            helper.make_node(
                "Unsqueeze", ["batch_value", "axes"], ["batch_vector"],
                name="unsqueeze_batch"
            ),
            helper.make_node(
                "Concat", ["batch_vector", "tail"], ["target_shape"],
                axis=0, name="concat_target"
            ),
            helper.make_node(
                "Reshape", ["x", "target_shape"], ["y"], name="dynamic_reshape"
            ),
        ]
        axes = initializer("axes", np.array([0], dtype=np.int64))
        return make_model(nodes, [x], [y], [indices, axes, tail])

    def test_continuous_dynamic_shape_execution_and_fixed_dimension_validation(self):
        model = self.make_model()
        stub = import_model(model)
        ort_session = (
            ort.InferenceSession(
                model.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            if ort is not None
            else None
        )
        self.assertEqual(stub.tensors["x_shape"].dtype(), TensorProto.INT64)
        for batch in (1, 2, 8, 3, 1):
            values = np.arange(batch * 6, dtype=np.float32).reshape(batch, 2, 3)
            stub.set_input([[batch, 2, 3]])
            stub.inputs["x"].copyin_numpy(values)
            stub.run()
            self.assertEqual(stub.getShape("y"), [batch, 6])
            actual = np.asarray(stub.outputs["y"].copyout_float()).reshape(batch, 6)
            np.testing.assert_allclose(actual, values.reshape(batch, 6), rtol=0, atol=0)
            if ort_session is not None:
                expected = ort_session.run(["y"], {"x": values})[0]
                self.assertEqual(tuple(expected.shape), (batch, 6))
                np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

        with self.assertRaisesRegex(ValueError, "fixed dimension 1 must be 2"):
            stub.set_input([[2, 4, 3]])

    def test_static_shape_subgraph_is_folded_but_dynamic_shape_is_retained(self):
        model = self.make_model()
        static_input = value_info("x", [2, 2, 3])
        model.graph.input[0].CopyFrom(static_input)
        nodes = list(model.graph.node)
        model.graph.ClearField("node")
        model.graph.node.extend(reversed(nodes))
        stub = import_model(model)
        self.assertEqual(stub.shape_optimization_stats["folded_static_shape_nodes"], 4)

    def test_dynamic_height_width_shape_chain_matches_ort(self):
        x = value_info("x", [1, 3, None, None])
        y = value_info("y", [1, 3, None, None])
        nodes = [
            helper.make_node("Shape", ["x"], ["x_shape"], name="shape"),
            helper.make_node(
                "Gather", ["x_shape", "height_index"], ["height"],
                axis=0, name="gather_height"
            ),
            helper.make_node(
                "Gather", ["x_shape", "width_index"], ["width"],
                axis=0, name="gather_width"
            ),
            helper.make_node(
                "Unsqueeze", ["height", "axes"], ["height_vector"],
                name="unsqueeze_height"
            ),
            helper.make_node(
                "Unsqueeze", ["width", "axes"], ["width_vector"],
                name="unsqueeze_width"
            ),
            helper.make_node(
                "Concat",
                ["batch", "channels", "height_vector", "width_vector"],
                ["target_shape"], axis=0, name="concat_target"
            ),
            helper.make_node(
                "Reshape", ["x", "target_shape"], ["y"], name="dynamic_reshape"
            ),
        ]
        initializers = [
            initializer("height_index", np.array(2, dtype=np.int64)),
            initializer("width_index", np.array(3, dtype=np.int64)),
            initializer("axes", np.array([0], dtype=np.int64)),
            initializer("batch", np.array([1], dtype=np.int64)),
            initializer("channels", np.array([3], dtype=np.int64)),
        ]
        model = make_model(nodes, [x], [y], initializers)
        stub = import_model(model)
        ort_session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        ) if ort is not None else None

        for height, width in ((2, 3), (5, 4), (1, 7), (3, 2), (2, 3)):
            values = np.arange(3 * height * width, dtype=np.float32).reshape(
                1, 3, height, width
            )
            stub.set_input([[1, 3, height, width]])
            stub.inputs["x"].copyin_numpy(values)
            stub.run()
            actual = np.asarray(stub.outputs["y"].copyout_float()).reshape(
                1, 3, height, width
            )
            np.testing.assert_array_equal(actual, values)
            if ort_session is not None:
                expected = ort_session.run(["y"], {"x": values})[0]
                np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

    def test_runtime_axes_for_unsqueeze_matches_ort(self):
        x = value_info("x", [2, 3])
        axes = value_info("axes", [1], TensorProto.INT64)
        y = value_info("y", [None, None, None])
        model = make_model(
            [helper.make_node("Unsqueeze", ["x", "axes"], ["y"])],
            [x, axes],
            [y],
        )
        stub = import_model(model)
        values = np.arange(6, dtype=np.float32).reshape(2, 3)
        stub.inputs["axes"].copyin_numpy(np.array([1], dtype=np.int64))
        stub.set_input([[2, 3], [1]])
        stub.inputs["x"].copyin_numpy(values)
        stub.run()
        self.assertEqual(stub.getShape("y"), [2, 1, 3])

        stub.inputs["axes"].copyin_numpy(np.array([0], dtype=np.int64))
        stub.set_input([[2, 3], [1]])
        stub.inputs["x"].copyin_numpy(values)
        stub.run()
        self.assertEqual(stub.getShape("y"), [1, 2, 3])
        actual = np.asarray(stub.outputs["y"].copyout_float()).reshape(1, 2, 3)
        np.testing.assert_array_equal(actual, values.reshape(1, 2, 3))

    def test_runtime_axes_for_squeeze_matches_ort(self):
        x = value_info("x", [1, 2, 1, 3])
        axes = value_info("axes", [1], TensorProto.INT64)
        y = value_info("y", [None, None, None])
        model = make_model(
            [helper.make_node("Squeeze", ["x", "axes"], ["y"])],
            [x, axes],
            [y],
        )
        stub = import_model(model)
        values = np.arange(6, dtype=np.float32).reshape(1, 2, 1, 3)
        stub.inputs["axes"].copyin_numpy(np.array([2], dtype=np.int64))
        stub.set_input([[1, 2, 1, 3], [1]])
        stub.inputs["x"].copyin_numpy(values)
        stub.run()
        self.assertEqual(stub.getShape("y"), [1, 2, 3])
        actual = np.asarray(stub.outputs["y"].copyout_float()).reshape(1, 2, 3)
        np.testing.assert_array_equal(actual, values.reshape(1, 2, 3))

    def test_dynamic_sequence_length_matches_ort(self):
        x = value_info("x", [1, None, 4])
        y = value_info("y", [1, None, 4])
        nodes = [
            helper.make_node("Shape", ["x"], ["shape"]),
            helper.make_node(
                "Gather", ["shape", "sequence_index"], ["sequence"], axis=0
            ),
            helper.make_node("Unsqueeze", ["sequence", "axes"], ["sequence_vector"]),
            helper.make_node(
                "Concat", ["batch", "sequence_vector", "hidden"],
                ["target"], axis=0
            ),
            helper.make_node("Reshape", ["x", "target"], ["reshaped"]),
            helper.make_node("MatMul", ["reshaped", "weight"], ["y"]),
        ]
        initializers = [
            initializer("sequence_index", np.array(1, dtype=np.int64)),
            initializer("axes", np.array([0], dtype=np.int64)),
            initializer("batch", np.array([1], dtype=np.int64)),
            initializer("hidden", np.array([4], dtype=np.int64)),
            initializer("weight", np.eye(4, dtype=np.float32)),
        ]
        model = make_model(nodes, [x], [y], initializers)
        stub = import_model(model)
        ort_session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        ) if ort is not None else None

        for sequence in (1, 7, 3, 12, 1):
            values = np.arange(sequence * 4, dtype=np.float32).reshape(1, sequence, 4)
            stub.set_input([[1, sequence, 4]])
            stub.inputs["x"].copyin_numpy(values)
            stub.run()
            self.assertEqual(stub.getShape("y"), [1, sequence, 4])
            actual = np.asarray(stub.outputs["y"].copyout_float()).reshape(
                1, sequence, 4
            )
            np.testing.assert_array_equal(actual, values)
            if ort_session is not None:
                expected = ort_session.run(["y"], {"x": values})[0]
                np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

    def test_dynamic_image_conv_matches_ort(self):
        x = value_info("x", [1, 1, None, None])
        y = value_info("y", [1, 1, None, None])
        weight = initializer(
            "weight", np.ones((1, 1, 3, 3), dtype=np.float32)
        )
        model = make_model(
            [
                helper.make_node(
                    "Conv", ["x", "weight"], ["conv"],
                    pads=[1, 1, 1, 1], name="conv"
                ),
                helper.make_node("Relu", ["conv"], ["y"], name="relu"),
            ],
            [x],
            [y],
            [weight],
        )
        stub = import_model(model)
        ort_session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        ) if ort is not None else None
        for height, width in ((5, 5), (7, 6), (3, 8), (5, 5)):
            values = np.arange(height * width, dtype=np.float32).reshape(
                1, 1, height, width
            )
            stub.set_input([[1, 1, height, width]])
            stub.inputs["x"].copyin_numpy(values)
            stub.run()
            actual = np.asarray(stub.outputs["y"].copyout_float()).reshape(
                1, 1, height, width
            )
            if ort_session is not None:
                expected = ort_session.run(["y"], {"x": values})[0]
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_symbolic_and_unknown_dimensions_are_distinguished(self):
        model = make_model(
            [helper.make_node("Identity", ["x"], ["y"])],
            [value_info("x", ["batch", None, 3])],
            [value_info("y", ["batch", None, 3])],
        )
        stub = import_model(model)
        self.assertEqual(
            stub._input_shape_specs["x"], [(None, "batch"), (None, None), (3, None)]
        )


if __name__ == "__main__":
    unittest.main()
