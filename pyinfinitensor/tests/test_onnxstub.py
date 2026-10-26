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
    def test_dynamic_template_does_not_run_simplifier_verification(self):
        model = make_model(
            [helper.make_node("Identity", ["x"], ["y"])],
            [value_info("x", ["B", 2])],
            [value_info("y", ["B", 2])],
        )
        with patch.object(
            onnx_frontend,
            "simplify",
            side_effect=AssertionError("unexpected simplifier forward"),
        ):
            stub = OnnxStub(model, backend.cpu_runtime())
        values = np.arange(6, dtype=np.float32).reshape(3, 2)
        np.testing.assert_array_equal(stub.run_numpy({"x": values})["y"], values)

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
                    actual = np.asarray(stub.outputs["y"].copyout_float()).reshape(3, 2)
                    np.testing.assert_allclose(actual, x @ weight)

    def test_dynamic_shape_parameter_reuses_weight_upload(self):
        # Shape -> Gather -> Unsqueeze -> Concat is representative of the
        # metadata subgraphs emitted by the BERT ONNX export.  No Shape kernel
        # is inserted into the executable graph.
        x = value_info("x", ["batch", "sequence"])
        y = value_info("y", ["batch", "sequence"])
        index = initializer("batch_index", [0], np.int64)
        sequence_index = initializer("sequence_index", [1], np.int64)
        axes = initializer("axes", [0], np.int64)
        nodes = [
            helper.make_node("Shape", ["x"], ["shape"]),
            helper.make_node("Gather", ["shape", "batch_index"], ["batch"], axis=0),
            helper.make_node(
                "Gather", ["shape", "sequence_index"], ["sequence"], axis=0
            ),
            helper.make_node("Unsqueeze", ["batch", "axes"], ["batch_vector"]),
            helper.make_node("Unsqueeze", ["sequence", "axes"], ["sequence_vector"]),
            helper.make_node(
                "Concat", ["batch_vector", "sequence_vector"], ["target"], axis=0
            ),
            helper.make_node("Reshape", ["x", "target"], ["y"]),
        ]
        stub = import_model(make_model(nodes, [x], [y], [index, sequence_index, axes]))
        uploaded = stub.weight_upload_count
        for batch, sequence in ((1, 2), (7, 17), (3, 511), (16, 128), (1, 17)):
            values = np.arange(batch * sequence, dtype=np.float32).reshape(
                batch, sequence
            )
            output = stub.run_numpy({"x": values})["y"]
            np.testing.assert_array_equal(output, values)
        self.assertEqual(stub.weight_upload_count, uploaded)


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
        inference_training = initializer("training", np.array(False, dtype=np.bool_))
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


class TestOnnxStubInfini(unittest.TestCase):
    @unittest.skipUnless(
        hasattr(backend, "InfiniRuntime"), "InfiniOps provider not built"
    )
    def test_dynamic_reallocation_restores_initializer(self):
        from pyinfinitensor.embedding import fp32_inference_policy

        # This checks allocator correctness against NumPy FP32, so vendor
        # reduced-precision matmul must not change the numerical oracle.
        with fp32_inference_policy():
            weight = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            model = make_model(
                [helper.make_node("MatMul", ["x", "weight"], ["y"])],
                [value_info("x", [1, 2])],
                [value_info("y", [1, 2])],
                [initializer("weight", weight)],
            )
            for use_naive_allocator in (False, True):
                with self.subTest(use_naive_allocator=use_naive_allocator):
                    stub = import_model(
                        model,
                        backend.runtime(backend.default_infini_device()),
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
                        np.testing.assert_allclose(
                            actual, x @ weight, rtol=1e-5, atol=1e-6
                        )
                    if not use_naive_allocator:
                        stub.trim_memory()
                        stub.run()
                        actual = np.asarray(stub.outputs["y"].copyout_float()).reshape(
                            2, 2
                        )
                        np.testing.assert_allclose(
                            actual, x @ weight, rtol=1e-5, atol=1e-6
                        )


class TestIntermediateShapeMetadata(unittest.TestCase):
    def test_variadic_shape_extrema_and_slice_optional_axes(self):
        evaluator = onnx_frontend.ShapeValueEvaluator(make_model([], [], []))
        values = {
            "a": np.array([2], np.int64),
            "b": np.array([3], np.int64),
            "c": np.array([10], np.int64),
        }
        maximum = evaluator._execute(
            helper.make_node("Max", ["a", "b", "c"], ["y"]), values, {}
        )[0]
        np.testing.assert_array_equal(maximum, [10])
        values["c"] = np.array([-10], np.int64)
        minimum = evaluator._execute(
            helper.make_node("Min", ["a", "b", "c"], ["y"]), values, {}
        )[0]
        np.testing.assert_array_equal(minimum, [-10])
        values = {
            "x": np.array([1, 2, 3, 4], np.int64),
            "start": np.array([0], np.int64),
            "end": np.array([4], np.int64),
            "step": np.array([2], np.int64),
        }
        sliced = evaluator._execute(
            helper.make_node("Slice", ["x", "start", "end", "", "step"], ["y"]),
            values,
            {},
        )[0]
        np.testing.assert_array_equal(sliced, [1, 3])

    def test_dynamic_slice_with_large_end_without_optional_parameters(self):
        model = make_model(
            [
                helper.make_node("Shape", ["x"], ["size"]),
                helper.make_node("Max", ["size", "limit"], ["end"]),
                helper.make_node("Slice", ["x", "start", "end"], ["y"]),
            ],
            [value_info("x", ["S"])],
            [value_info("y", ["S"])],
            [
                initializer("start", np.array([0], np.int64)),
                initializer("limit", np.array([np.iinfo(np.int64).max], np.int64)),
            ],
        )
        stub = import_model(model)
        for length in (1, 7, 2):
            values = np.arange(length, dtype=np.float32)
            stub.prepare({"x": values.shape})
            self.assertEqual(stub.outputs["y"].shape(), list(values.shape))

    def test_constant_slice_endpoints_follow_current_intermediate_shape(self):
        model = make_model(
            [
                helper.make_node("Identity", ["x"], ["intermediate"]),
                helper.make_node("Slice", ["intermediate", "starts", "ends"], ["y"]),
            ],
            [value_info("x", ["S"])],
            [value_info("y", [None])],
            [
                initializer("starts", np.array([0], np.int64)),
                initializer("ends", np.array([2], np.int64)),
            ],
        )
        stub = import_model(model)
        for length in (1, 3, 2, 1):
            values = np.arange(length, dtype=np.float32)
            stub.prepare({"x": values.shape})
            self.assertEqual(stub.outputs["y"].shape(), list(values[:2].shape))

    def test_non_integer_shapes_are_rejected_without_mutation(self):
        model = make_model(
            [helper.make_node("Identity", ["x"], ["y"])],
            [value_info("x", ["B", 2])],
            [value_info("y", ["B", 2])],
        )
        stub = import_model(model)
        value = np.array([[2.0, 3.0]], np.float32)
        np.testing.assert_array_equal(stub.run_numpy({"x": value})["y"], value)
        for invalid in ((1.5, 2), (True, 2), ("1", 2)):
            with (
                self.subTest(shape=invalid),
                self.assertRaisesRegex(ValueError, "invalid shape"),
            ):
                stub.prepare({"x": invalid})
        np.testing.assert_array_equal(stub.run_numpy({"x": value})["y"], value)

    def test_injected_allocation_failure_rolls_back(self):
        model = make_model(
            [helper.make_node("Identity", ["x"], ["y"])],
            [value_info("x", ["B", 2])],
            [value_info("y", ["B", 2])],
        )
        stub = import_model(model)
        value = np.array([[2.0, 3.0]], np.float32)
        np.testing.assert_array_equal(stub.run_numpy({"x": value})["y"], value)
        real_handler = stub.handler
        proxy = Mock(wraps=real_handler)
        calls = 0

        def fail_once(use_naive):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("injected allocation failure")
            return real_handler.data_malloc(use_naive)

        proxy.data_malloc.side_effect = fail_once
        stub.handler = proxy
        with self.assertRaisesRegex(RuntimeError, "injected allocation failure"):
            stub.prepare({"x": (3, 2)})
        self.assertEqual(calls, 2)
        np.testing.assert_array_equal(stub.run_numpy({"x": value})["y"], value)

    def test_integer_division_and_negative_unsqueeze_axes(self):
        evaluator = onnx_frontend.ShapeValueEvaluator(make_model([], [], []))
        values = {
            "a": np.array([-7, 7, -7, 2**60 + 1], np.int64),
            "b": np.array([3, -3, -3, 3], np.int64),
        }
        out = evaluator._execute(
            helper.make_node("Div", ["a", "b"], ["y"]), values, {}
        )[0]
        np.testing.assert_array_equal(out, [-2, -2, 2, (2**60 + 1) // 3])
        values = {"x": np.ones((2, 3), np.int64), "axes": np.array([-1, -3])}
        out = evaluator._execute(
            helper.make_node("Unsqueeze", ["x", "axes"], ["y"]), values, {}
        )[0]
        self.assertEqual(out.shape, (2, 1, 3, 1))

    def test_embedding_shape_does_not_evaluate_or_remove_embedding(self):
        model = make_model(
            [
                helper.make_node(
                    "Gather", ["w", "ids"], ["embedded"], name="embedding"
                ),
                helper.make_node("Shape", ["embedded"], ["shape"]),
                helper.make_node("Gather", ["shape", "zero"], ["b"]),
                helper.make_node("Unsqueeze", ["b", "axes"], ["bv"]),
                helper.make_node("Concat", ["bv", "tail"], ["target"], axis=0),
                helper.make_node("Reshape", ["embedded", "target"], ["y"]),
            ],
            [value_info("ids", ["B", "S"], TensorProto.INT64)],
            [value_info("y", ["B", "T"])],
            [
                initializer("w", np.ones((10, 4), np.float32)),
                initializer("zero", np.array(0, np.int64)),
                initializer("axes", np.array([0], np.int64)),
                initializer("tail", np.array([-1], np.int64)),
            ],
        )
        evaluator = onnx_frontend.ShapeValueEvaluator(model)
        self.assertNotIn(0, onnx_frontend._shape_control_node_indexes(model))
        for b, s in [(1, 17), (7, 83), (3, 511), (1, 17)]:
            evaluator.evaluate({"ids": (b, s)})
            self.assertEqual(evaluator.value("target"), [b, -1])
            self.assertEqual(evaluator.shapes["embedded"], (b, s, 4))
            self.assertEqual(evaluator.shapes["y"], (b, s * 4))
            self.assertNotIn("embedded", evaluator._values)


if __name__ == "__main__":
    unittest.main()
