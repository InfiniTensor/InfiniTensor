import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
from onnx import TensorProto, checker, helper, numpy_helper, save_model
from onnx.external_data_helper import set_external_data, uses_external_data

from pyinfinitensor import backend
from pyinfinitensor import onnx as onnx_frontend
from pyinfinitensor.external_weights import read_external_array
from pyinfinitensor.onnx import OnnxStub, OnnxWeightStore, _parse_attribute
from pyinfinitensor.onnx_rewrites import fuse_fp32_erf_gelu, fuse_fp32_rms_norm, fuse_masked_scatter


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
    def _check_live_conv_bias(self, runtime, np_dtype, onnx_dtype):
        data_shape = [1, 2, 2, 2]
        model = make_model(
            [helper.make_node("Conv", ["x", "w", "b"], ["y"], group=2)],
            [value_info("x", data_shape, onnx_dtype), value_info("b", [2], onnx_dtype)],
            [value_info("y", data_shape, onnx_dtype)],
            [initializer("w", np.array([2, -1], dtype=np_dtype).reshape(2, 1, 1, 1))],
        )
        stub = import_model(model, runtime=runtime)
        ops = stub.handler.operators()
        expected_ops = [backend.OpTypeId.Conv]
        if onnx_dtype == TensorProto.FLOAT16:
            expected_ops.extend((backend.OpTypeId.Reshape, backend.OpTypeId.Add))
        self.assertEqual([op.op_type().id() for op in ops], expected_ops)
        self.assertEqual(len(ops[0].inputs()), 2 if onnx_dtype == TensorProto.FLOAT16 else 3)
        data = np.arange(8, dtype=np_dtype).reshape(data_shape)
        scale = np.array([2, -1], dtype=np_dtype).reshape(1, 2, 1, 1)
        stub.inputs["x"].copyin_numpy(data)
        for offsets in ([4, 7], [-3, 1]):
            bias = np.array(offsets, dtype=np_dtype)
            stub.inputs["b"].copyin_numpy(bias)
            stub.run()
            np.testing.assert_array_equal(
                stub.outputs["y"].copyout_numpy(), data * scale + bias.reshape(1, 2, 1, 1)
            )
        exported = stub.to_onnx("conv_with_bias")
        checker.check_model(exported)
        self.assertEqual([len(node.input) for node in exported.graph.node if node.op_type == "Conv"],
                         [2 if onnx_dtype == TensorProto.FLOAT16 else 3])

    def test_cpu_conv_keeps_live_bias(self):
        self._check_live_conv_bias(backend.cpu_runtime(), np.float32, TensorProto.FLOAT)

    def test_cambricon_conv_keeps_live_bias(self):
        try:
            runtime = backend.runtime("cambricon", 0)
        except (RuntimeError, AttributeError) as error:
            self.skipTest(str(error))
        for np_dtype, onnx_dtype in (
            (np.float32, TensorProto.FLOAT), (np.float16, TensorProto.FLOAT16),
        ):
            with self.subTest(dtype=np_dtype):
                self._check_live_conv_bias(runtime, np_dtype, onnx_dtype)

    def test_cambricon_fp16_conv_preshapes_only_constant_bias(self):
        try:
            runtime = backend.runtime("cambricon", 0)
        except (RuntimeError, AttributeError) as error:
            self.skipTest(str(error))
        shape = [1, 2, 2, 2]
        model = make_model(
            [helper.make_node("Conv", ["x", "w", "b"], ["y"], group=2)],
            [value_info("x", shape, TensorProto.FLOAT16)],
            [value_info("y", shape, TensorProto.FLOAT16)],
            [initializer("w", np.array([2, -1], np.float16).reshape(2, 1, 1, 1)),
             initializer("b", [4, 7], np.float16)],
        )
        stub = import_model(model, runtime=runtime)
        self.assertEqual([op.op_type().id() for op in stub.handler.operators()],
                         [backend.OpTypeId.Conv, backend.OpTypeId.Add])
        data = np.arange(8, dtype=np.float16).reshape(shape)
        stub.inputs["x"].copyin_numpy(data)
        stub.run()
        expected = data * np.array([2, -1], np.float16).reshape(1, 2, 1, 1)
        expected += np.array([4, 7], np.float16).reshape(1, 2, 1, 1)
        np.testing.assert_array_equal(stub.outputs["y"].copyout_numpy(), expected)
        checker.check_model(stub.to_onnx("conv_with_constant_fp16_bias"))

    def test_cambricon_conv_layout_preserves_live_inputs_and_public_outputs(self):
        try:
            runtime = backend.runtime("cambricon", 0)
        except (RuntimeError, AttributeError) as error:
            self.skipTest(str(error))
        if not backend.GraphHandler(runtime).supports_channels_last_conv():
            self.skipTest("Requires the channels-last Cambricon backend")

        for dtype, np_dtype in ((TensorProto.FLOAT, np.float32), (TensorProto.FLOAT16, np.float16)):
            with self.subTest(dtype=dtype):
                shape = [2, 4, 3, 5]
                model = make_model(
                    [helper.make_node("Conv", ["x", "w", "b"], ["conv"], group=2),
                     helper.make_node("Sigmoid", ["conv"], ["gate"]),
                     helper.make_node("Mul", ["gate", "conv"], ["act"]),
                     helper.make_node("Conv", ["act", "w"], ["y"], group=2)],
                    [value_info("x", shape, dtype), value_info("b", [4], dtype)],
                    [value_info("conv", shape, dtype), value_info("y", shape, dtype)],
                    [initializer("w", np.linspace(-.4, .5, 8).reshape(4, 2, 1, 1), np_dtype)],
                )
                baseline = OnnxStub(model, runtime, simplify_model=False)
                configurations = [("channels_last", False), ("channels_last_silu", False)]
                if dtype == TensorProto.FLOAT16:
                    configurations.append(("channels_last_silu", True))
                for mode, fused_bias in configurations:
                    optimized = OnnxStub(model, runtime, simplify_model=False,
                                         conv_layout=mode, fuse_fp16_conv_bias=fused_bias)
                    checker.check_model(optimized.to_onnx("layout_round_trip"))
                    ops = optimized.handler.operators()
                    convs = [op for op in ops if op.op_type().id() == backend.OpTypeId.Conv]
                    self.assertTrue(all(backend.conv_channels_last_of(op) for op in convs))
                    self.assertEqual(len(convs), 2)
                    self.assertEqual(len(convs[0].inputs()), 3 if fused_bias or dtype == TensorProto.FLOAT else 2)
                    self.assertEqual(sum(op.op_type().id() == backend.OpTypeId.Silu for op in ops),
                                     int(mode.endswith("_silu")))
                    for step in (0, 1):
                        pixels = np.linspace(-2, 3 + step, np.prod(shape)).reshape(shape).astype(np_dtype)
                        bias = np.array([.1, -.2, .3, .4 + step], dtype=np_dtype)
                        for stub in (baseline, optimized):
                            stub.inputs["x"].copyin_numpy(pixels)
                            stub.inputs["b"].copyin_numpy(bias)
                            stub.run_with_graph() if stub is optimized else stub.run()
                        for name in ("conv", "y"):
                            actual = optimized.outputs[name].copyout_numpy()
                            expected = baseline.outputs[name].copyout_numpy()
                            if mode == "channels_last":
                                np.testing.assert_array_equal(actual, expected)
                            elif dtype == TensorProto.FLOAT16:
                                np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=2e-3)
                            else:
                                np.testing.assert_allclose(actual, expected, atol=2e-6)

    def test_weight_store_reuses_only_identical_immutable_initializers(self):
        runtime = backend.cpu_runtime()
        store = OnnxWeightStore(runtime)
        def build(values, naive=False):
            model = make_model(
                [helper.make_node("Add", ["x", "weight"], ["y"])],
                [value_info("x", ["batch", 2])], [value_info("y", ["batch", 2])],
                [initializer("weight", values, np.float32)],
            )
            return OnnxStub(model, runtime, simplify_model=False, weight_store=store,
                            input_shapes={"x": [2, 2]}, use_naive_allocator=naive)

        first = build([1, 2])
        same = build([1, 2], naive=True)
        different = build([3, 4])
        self.assertEqual(same.shared_weight_names, {"weight"})
        self.assertEqual(same.shared_weight_bytes, 8)
        self.assertEqual(different.shared_weight_names, set())
        with self.assertRaisesRegex(RuntimeError, "overwrite shared weight"):
            first.tensors["weight"].copyin_numpy(np.zeros(2, np.float32))
        del first, store
        same.init()
        for stub, values in ((same, [1, 2]), (different, [3, 4])):
            stub.inputs["x"].copyin_numpy(np.zeros((2, 2), np.float32))
            stub.run()
            np.testing.assert_array_equal(stub.outputs["y"].copyout_numpy(), [values, values])

    def test_batch_copy_validates_all_pairs_before_writing(self):
        try:
            runtime = backend.runtime("cambricon", 0)
        except (RuntimeError, AttributeError):
            runtime = backend.cpu_runtime()
        handler = backend.GraphHandler(runtime)
        left, right, first, second = [handler.tensor([2], TensorProto.FLOAT) for _ in range(4)]
        wrong = handler.tensor([1], TensorProto.FLOAT)
        handler.data_malloc()
        for tensor, values in ((left, [1, 2]), (right, [3, 4]),
                               (first, [0, 0]), (second, [0, 0]), (wrong, [8])):
            tensor.copyin_numpy(np.asarray(values, dtype=np.float32))
        with self.assertRaisesRegex(RuntimeError, "shape or dtype mismatch"):
            backend.copy_tensor_batch([(first, left), (second, wrong)])
        np.testing.assert_array_equal(first.copyout_numpy(), [0, 0])
        backend.copy_tensor_batch([(first, left), (second, right), (left, left)])
        np.testing.assert_array_equal(first.copyout_numpy(), [1, 2])
        np.testing.assert_array_equal(second.copyout_numpy(), [3, 4])

    def test_flatten_accepts_axis_equal_to_rank_and_scalar(self):
        for shape, axis, output_shape in (
            ([2, 3], 2, [6, 1]), ([2, 3], -2, [1, 6]), ([], 0, [1, 1]),
        ):
            with self.subTest(shape=shape, axis=axis):
                model = make_model(
                    [helper.make_node("Flatten", ["x"], ["y"], axis=axis)],
                    [value_info("x", shape)], [value_info("y", output_shape)],
                )
                stub = OnnxStub(model, backend.cpu_runtime(), simplify_model=False)
                values = np.arange(max(1, int(np.prod(shape))), dtype=np.float32).reshape(shape)
                stub.inputs["x"].copyin_numpy(values)
                stub.run()
                np.testing.assert_array_equal(
                    stub.outputs["y"].copyout_numpy(), values.reshape(output_shape)
                )

    def test_gather_wraps_live_negative_indices_on_cambricon(self):
        try:
            runtime = backend.runtime("cambricon", 0)
        except (RuntimeError, AttributeError) as error:
            self.skipTest(str(error))
        for dtype, shape, onnx_dtype, index_cases in (
            (np.int32, [], TensorProto.INT32, (-1, -3, 1)),
            (np.int64, [2, 2], TensorProto.INT64,
             ([[-3, -1], [0, 2]], [[2, -2], [-1, 1]])),
        ):
            with self.subTest(index_dtype=dtype, index_shape=shape):
                model = make_model(
                    [helper.make_node("Gather", ["x", "indices"], ["y"], axis=-1)],
                    [value_info("x", [2, 3]), value_info("indices", shape, onnx_dtype)],
                    [value_info("y", [2, *shape])],
                )
                stub = OnnxStub(model, runtime, simplify_model=False)
                values = np.arange(6, dtype=np.float32).reshape(2, 3)
                stub.inputs["x"].copyin_numpy(values)
                for index in index_cases:
                    index = np.asarray(index, dtype=dtype)
                    stub.inputs["indices"].copyin_numpy(index)
                    stub.run()
                    np.testing.assert_array_equal(
                        stub.outputs["y"].copyout_numpy(), np.take(values, index, axis=-1)
                    )

    def test_copy_from_reuses_tensor_data_without_numpy(self):
        try:
            runtime = backend.runtime("cambricon", 0)
        except (RuntimeError, AttributeError):
            runtime = backend.cpu_runtime()
        model = make_model(
            [helper.make_node("Identity", ["x"], ["y"])],
            [value_info("x", [2, 3])], [value_info("y", [2, 3])],
        )
        first = OnnxStub(model, runtime, simplify_model=False)
        second = OnnxStub(model, runtime, simplify_model=False)
        values = np.arange(6, dtype=np.float32).reshape(2, 3)
        first.inputs["x"].copyin_numpy(values)
        first.run()
        kept = first.clone_KV(first.outputs["y"])
        del first
        second.inputs["x"].copy_from(kept)
        second.run()
        np.testing.assert_array_equal(second.outputs["y"].copyout_numpy(), values)

    def test_stable_half_rms_norm_does_not_overflow_large_inputs(self):
        try:
            runtime = backend.runtime("cambricon", 0)
        except (RuntimeError, AttributeError) as error:
            self.skipTest(str(error))
        model = make_model(
            [helper.make_node("RMSNorm", ["x", "weight"], ["y"], domain="infinitensor",
                              epsilon=1e-6, stable_half=1)],
            [value_info("x", [2, 4], TensorProto.FLOAT16)],
            [value_info("y", [2, 4], TensorProto.FLOAT16)],
            [initializer("weight", [1, 2, 1, 2], np.float16)], check=False,
        )
        model.opset_import.append(helper.make_opsetid("infinitensor", 1))
        stub = OnnxStub(model, runtime, simplify_model=False)
        values = np.array([[2048, -2048, 4096, -4096], [0, .01, -.02, .03]], np.float16)
        stub.inputs["x"].copyin_numpy(values)
        stub.run()
        actual = stub.outputs["y"].copyout_numpy()
        source = values.astype(np.float32)
        expected = source / np.sqrt((source * source).mean(axis=-1, keepdims=True) + 1e-6)
        expected *= np.array([1, 2, 1, 2], np.float32)
        self.assertEqual(actual.dtype, np.float16)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, expected, rtol=.003, atol=.003)

    @staticmethod
    def masked_scatter_model(start=0):
        return make_model(
            [
                helper.make_node("NonZero", ["mask"], ["nonzero"]),
                helper.make_node("Transpose", ["nonzero"], ["coords"], perm=[1, 0]),
                helper.make_node("Shape", ["coords"], ["coords_shape"]),
                helper.make_node("Gather", ["coords_shape", "zero_scalar"], ["count"]),
                helper.make_node("Unsqueeze", ["count", "zero"], ["end"]),
                helper.make_node("Reshape", ["source", "flat"], ["flattened"]),
                helper.make_node("Slice", ["flattened", "start", "end", "zero"], ["updates"]),
                helper.make_node("ScatterND", ["x", "coords", "updates"], ["y"]),
            ],
            [value_info("x", [2, 3]), value_info("mask", [2, 3], TensorProto.BOOL),
             value_info("source", [2, 2])],
            [value_info("y", [2, 3])],
            [initializer("zero_scalar", 0, np.int64), initializer("zero", [0], np.int64),
             initializer("flat", [-1], np.int64), initializer("start", [start], np.int64)],
        )

    def test_masked_scatter_fusion_checks_slice_and_preserves_shared_coordinates(self):
        model = self.masked_scatter_model(start=1)
        self.assertEqual(fuse_masked_scatter(model), 0)
        model = self.masked_scatter_model()
        model.graph.output.append(value_info("coords", ["count", 2], TensorProto.INT64))
        self.assertEqual(fuse_masked_scatter(model), 1)
        self.assertEqual([node.op_type for node in model.graph.node],
                         ["NonZero", "Transpose", "MaskedScatter"])

    def test_masked_scatter_runs_different_masks_on_cambricon(self):
        try:
            runtime = backend.runtime("cambricon", 0)
        except (RuntimeError, AttributeError) as error:
            self.skipTest(str(error))
        stub = OnnxStub(self.masked_scatter_model(), runtime, simplify_model=False)
        data = np.arange(6, dtype=np.float32).reshape(2, 3)
        source = np.array([[9, 8], [7, 6]], np.float32)
        for mask in (np.array([[True, False, True], [False, False, True]]),
                     np.array([[False, False, False], [True, False, True]])):
            expected = data.copy()
            expected[mask] = source.reshape(-1)[:mask.sum()]
            for name, array in {"x": data, "mask": mask, "source": source}.items():
                stub.inputs[name].copyin_numpy(array)
            stub.run()
            np.testing.assert_array_equal(stub.outputs["y"].copyout_numpy(), expected)

    def test_external_weights_remain_lazy_and_use_model_directory(self):
        weight = np.arange(65537, dtype=np.float32)
        model = make_model(
            [helper.make_node("Add", ["x", "weight"], ["y"])],
            [value_info("x", weight.shape)], [value_info("y", weight.shape)],
            [initializer("weight", weight)],
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            save_model(model, path, save_as_external_data=True,
                       location=f"{path.parent.name}-weights.bin", size_threshold=0)
            stub = OnnxStub(path, backend.cpu_runtime())
            self.assertTrue(uses_external_data(stub._initializer_by_name["weight"]))
            self.assertFalse(stub._initializer_by_name["weight"].HasField("raw_data"))
            stub.inputs["x"].copyin_numpy(np.ones_like(weight))
            stub.run()
            np.testing.assert_array_equal(stub.outputs["y"].copyout_numpy(), weight + 1)

    @unittest.skipUnless(hasattr(checker.C, "_resolve_external_data_location"),
                         "ONNX external data resolver unavailable")
    def test_external_array_reads_only_the_declared_range(self):
        weight = np.arange(65537, dtype=np.float32)
        tensor = initializer("weight", weight)
        prefix = b"unrelated bytes"
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            payload = folder / "weights.bin"
            payload.write_bytes(prefix + weight.tobytes() + b"other weights")
            set_external_data(tensor, location=payload.name, offset=len(prefix),
                              length=weight.nbytes)
            tensor.ClearField("raw_data")
            np.testing.assert_array_equal(read_external_array(tensor, directory), weight)
            self.assertFalse(tensor.HasField("raw_data"))
            payload.write_bytes(prefix + weight.tobytes()[:-1])
            with self.assertRaisesRegex(ValueError, "offset or length"):
                read_external_array(tensor, directory)

    def test_rmsnorm_fusion_requires_last_axis_and_private_intermediates(self):
        nodes = [
            helper.make_node("Pow", ["x", "two"], ["square"]),
            helper.make_node("ReduceMean", ["square", "axis"], ["mean"], keepdims=1),
            helper.make_node("Add", ["mean", "eps"], ["shifted"]),
            helper.make_node("Sqrt", ["shifted"], ["root"]),
            helper.make_node("Div", ["one", "root"], ["inverse"]),
            helper.make_node("Mul", ["x", "inverse"], ["normalized"]),
            helper.make_node("Mul", ["normalized", "weight"], ["y"]),
        ]
        for axis, public_mean, expected in [(-1, False, 1), (0, False, 0), (-1, True, 0)]:
            with self.subTest(axis=axis, public_mean=public_mean):
                outputs = [value_info("y", [2, 3])]
                if public_mean:
                    outputs.append(value_info("mean", [2, 1]))
                model = make_model(nodes, [value_info("x", [2, 3])], outputs, [
                    initializer("two", 2, np.float32), initializer("one", 1, np.float32),
                    initializer("eps", 1e-5, np.float32), initializer("axis", [axis], np.int64),
                    initializer("weight", [1, 2, 3], np.float32),
                ])
                self.assertEqual(fuse_fp32_rms_norm(model), expected)
                if expected:
                    self.assertEqual([node.op_type for node in model.graph.node], ["RMSNorm"])
                    self.assertEqual(list(model.graph.node[0].input), ["x", "weight"])

    def test_exact_gelu_fusion_preserves_public_intermediate(self):
        nodes = [
            helper.make_node("Div", ["x", "sqrt_two"], ["scaled"]),
            helper.make_node("Erf", ["scaled"], ["erf"]),
            helper.make_node("Add", ["one", "erf"], ["gate"]),
            helper.make_node("Mul", ["x", "gate"], ["product"]),
            helper.make_node("Mul", ["half", "product"], ["y"]),
        ]
        constants = [initializer("sqrt_two", 2**.5, np.float32),
                     initializer("one", 1, np.float32), initializer("half", .5, np.float32)]
        for visible in (False, True):
            outputs = [value_info("y", [2, 3])]
            if visible:
                outputs.append(value_info("erf", [2, 3]))
            model = make_model(nodes, [value_info("x", [2, 3])], outputs, constants)
            self.assertEqual(fuse_fp32_erf_gelu(model), 0 if visible else 1)
            if not visible:
                self.assertEqual([node.op_type for node in model.graph.node], ["Gelu"])

    def test_static_shape_subgraph_keeps_tensor_input_live(self):
        model = make_model(
            [
                helper.make_node("Shape", ["x"], ["dims"]),
                helper.make_node("Gather", ["dims", "index"], ["selected"], axis=0),
                helper.make_node("Concat", ["selected", "rest"], ["target"], axis=0),
                helper.make_node("Reshape", ["x", "target"], ["y"]),
            ],
            [value_info("x", ["batch", 2, 2])],
            [value_info("y", ["batch", 4])],
            [initializer("index", [0], np.int64), initializer("rest", [4], np.int64)],
        )
        with self.assertRaisesRegex(ValueError, "concrete ONNX input"):
            OnnxStub(model, backend.cpu_runtime(), simplify_model=False,
                     fold_static_shapes=True)
        for batch in (1, 3):
            stub = OnnxStub(model, backend.cpu_runtime(), simplify_model=False,
                            input_shapes={"x": [batch, 2, 2]}, fold_static_shapes=True)
            for shift in (0, 10):
                data = (np.arange(batch * 4, dtype=np.float32) + shift).reshape(batch, 2, 2)
                stub.inputs["x"].copyin_numpy(data)
                stub.run()
                np.testing.assert_array_equal(stub.outputs["y"].copyout_numpy(),
                                              data.reshape(batch, 4))
            with self.assertRaisesRegex(ValueError, "Reimport"):
                stub.set_input([[batch + 1, 2, 2]])

    def test_internal_copies_are_elided_without_aliasing_public_outputs(self):
        model = make_model([
            helper.make_node("Identity", ["x"], ["internal"]),
            helper.make_node("Cast", ["internal"], ["same_type"], to=TensorProto.FLOAT),
            helper.make_node("Reshape", ["same_type", "same_shape"], ["reshaped"]),
            helper.make_node("Identity", ["reshaped"], ["y"]),
        ], [value_info("x", [2, 3])], [value_info("y", [2, 3])],
           [initializer("same_shape", [2, 3], np.int64)])
        stub = OnnxStub(model, backend.cpu_runtime(), simplify_model=False,
                        fold_static_shapes=True, elide_internal_copies=True)
        self.assertEqual(stub.elided_internal_copies, 3)
        self.assertEqual(len(stub.handler.operators()), 1)
        self.assertNotEqual(stub.inputs["x"].fuid(), stub.outputs["y"].fuid())
        for shift in (0, 5):
            value = np.arange(6, dtype=np.float32).reshape(2, 3) + shift
            stub.inputs["x"].copyin_numpy(value)
            stub.run()
            np.testing.assert_array_equal(stub.outputs["y"].copyout_numpy(), value)

    def test_concrete_input_shape_resolves_symbolic_batch(self):
        model = make_model(
            [helper.make_node("Relu", ["x"], ["y"])],
            [value_info("x", ["batch", 2])],
            [value_info("y", ["batch", 2])],
        )
        before = model.SerializeToString()
        data = np.arange(6, dtype=np.float32).reshape(3, 2) - 2
        stub = OnnxStub(model, backend.cpu_runtime(), simplify_model=False,
                        input_shapes={"x": data.shape})
        stub.inputs["x"].copyin_numpy(data)
        stub.run()
        np.testing.assert_array_equal(stub.outputs["y"].copyout_numpy(), np.maximum(data, 0))
        self.assertEqual(model.SerializeToString(), before)
        for bad in ({"missing": [3, 2]}, {"x": [3]}, {"x": [3, 4]},
                    {"x": [-1, 2]}, {"x": [True, 2]}, {"x": [3.5, 2]}):
            with self.subTest(input_shapes=bad), self.assertRaises(ValueError):
                OnnxStub(model, backend.cpu_runtime(), simplify_model=False,
                         input_shapes=bad)

    def test_simplification_can_be_disabled_without_changing_results(self):
        model = make_model(
            [helper.make_node("Relu", ["x"], ["y"])],
            [value_info("x", [2])],
            [value_info("y", [2])],
        )
        original = model.SerializeToString()
        inputs = np.array([-2.0, 3.0], dtype=np.float32)
        for options, env in (({"simplify_model": False}, "0"), ({}, "1")):
            with self.subTest(options=options), patch.dict(
                os.environ, {"INFINITENSOR_SKIP_ONNXSIM": env}
            ), patch.object(onnx_frontend, "simplify") as simplify:
                stub = OnnxStub(model, backend.cpu_runtime(), **options)
                simplify.assert_not_called()
                stub.inputs["x"].copyin_numpy(inputs)
                stub.run()
                np.testing.assert_array_equal(stub.outputs["y"].copyout_float(), [0, 3])
                self.assertEqual(model.SerializeToString(), original)

    def test_explicit_simplification_overrides_environment(self):
        model = make_model(
            [helper.make_node("Relu", ["x"], ["y"])],
            [value_info("x", [2])],
            [value_info("y", [2])],
        )
        with patch.dict(os.environ, {"INFINITENSOR_SKIP_ONNXSIM": "1"}), patch.object(
            onnx_frontend, "simplify", side_effect=lambda candidate: (candidate, True)
        ) as simplify:
            OnnxStub(model, backend.cpu_runtime(), simplify_model=True)
            simplify.assert_called_once()

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
                    np.testing.assert_allclose(actual, x @ weight, rtol=1e-5, atol=1e-6)
                if not use_naive_allocator:
                    stub.trim_memory()
                    stub.run()
                    actual = np.asarray(stub.outputs["y"].copyout_float()).reshape(
                        2, 2
                    )
                    np.testing.assert_allclose(actual, x @ weight, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
