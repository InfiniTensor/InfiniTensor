"""ONNX execution regressions for the native Cambricon provider."""

import unittest

import numpy as np
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from pyinfinitensor import backend
from pyinfinitensor.onnx import OnnxStub


def cases():
    x = np.arange(6, dtype=np.float32).reshape(2, 3) - 2
    y = np.arange(6, dtype=np.float32).reshape(2, 3)
    z = np.arange(12, dtype=np.float32).reshape(3, 4)
    return {
        "add_fp32": ("Add", {"x": x, "y": y}, {}, {}),
        "add_fp16": ("Add", {"x": x.astype(np.float16), "y": y.astype(np.float16)}, {}, {}),
        "add_two_way_broadcast": ("Add", {"x": x[:, :1].copy(), "y": y[0].copy()}, {}, {}),
        "add_int64": ("Add", {"x": np.array([[2**40], [-2**40]], np.int64),
                              "y": np.array([2, -3], np.int64)}, {}, {}),
        "sub_broadcast": ("Sub", {"x": x, "y": y[0].copy()}, {}, {}),
        "sub_int64": ("Sub", {"x": np.array([[2**60], [-2**60]], np.int64),
                              "y": np.array([2**54 + 1, -3], np.int64)}, {}, {}),
        "div_broadcast": ("Div", {"x": x, "y": np.arange(1, 4, dtype=np.float32)}, {}, {}),
        "div_fp16_scalar": ("Div", {"x": x.astype(np.float16),
                                     "y": np.array(3, np.float16)}, {}, {}),
        "equal_int64": ("Equal", {"x": np.array([[2**60], [-2**60]], np.int64),
                                  "y": np.array([2**60, -2**60, 0], np.int64)}, {}, {}),
        "less_fp16": ("Less", {"x": x.astype(np.float16),
                               "y": np.array(0.5, np.float16)}, {}, {}),
        "tile_int64": ("Tile", {"x": np.array([[2**60], [-2**60]], np.int64)},
                         {"repeats": np.array([2, 3], np.int64)}, {}),
        "matmul": ("MatMul", {"x": x, "z": z}, {}, {}),
        "matmul_fp16": ("MatMul", {"x": (x / 3).astype(np.float16),
                                    "z": (z / 7).astype(np.float16)}, {}, {}),
        "matmul_batch_broadcast": (
            "MatMul",
            {"x": np.arange(12, dtype=np.float32).reshape(2, 1, 2, 3) / 5,
             "z": np.arange(36, dtype=np.float32).reshape(1, 3, 3, 4) / 7},
            {}, {},
        ),
        "matmul_batch_mixed_rank": (
            "MatMul",
            {"x": np.arange(24, dtype=np.float32).reshape(2, 2, 2, 3) / 5,
             "z": z / 7},
            {}, {},
        ),
        "matmul_singleton_batch": ("MatMul", {"x": x[None, ...], "z": z}, {}, {}),
        "matmul_transpose_bias": ("Gemm", {"x": x.T.copy(), "z": z.T.copy()},
                                   {"bias": np.arange(4, dtype=np.float32)},
                                   {"transA": 1, "transB": 1}),
        "softmax": ("Softmax", {"x": x}, {}, {"axis": -1}),
        "softmax_fp16_axis0": ("Softmax", {"x": x.astype(np.float16)}, {}, {"axis": 0}),
        "softmax_rank4": ("Softmax", {"x": np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5) / 12}, {}, {"axis": 2}),
        "gather": ("Gather", {"x": x}, {"indices": np.array([2, 0], np.int64)}, {"axis": 1}),
        "gather_negative_int64": ("Gather", {"x": x, "indices": np.array([-3, 0, -1], np.int64)},
                                  {}, {"axis": 1}),
        "gather_negative_scalar_int32": ("Gather", {"x": x, "indices": np.array(-1, np.int32)},
                                         {}, {"axis": 0}),
        "gather_nd_indices": ("Gather", {"x": x.astype(np.float16),
                                       "indices": np.array([[0, 2], [2, 1]], np.int64)},
                              {}, {"axis": -1}),
        "gather_scalar": ("Gather", {"x": x, "indices": np.array(1, np.int32)},
                          {}, {"axis": 0}),
        "cast_bool": ("Cast", {"x": x}, {}, {"to": TensorProto.BOOL}),
        "cast_identity_int64": ("Cast", {"x": np.array([-2**60, 2**60], np.int64)},
                                {}, {"to": TensorProto.INT64}),
        "cast_int64_bool": ("Cast", {"x": np.array([-2**40, -1, 0, 2**40], np.int64)},
                            {}, {"to": TensorProto.BOOL}),
        "cast_bool_fp16": ("Cast", {"x": x > 0}, {}, {"to": TensorProto.FLOAT16}),
        "cast_fp32_fp16": ("Cast", {"x": x / 3}, {}, {"to": TensorProto.FLOAT16}),
        "where": ("Where", {"mask": x > 0, "x": x, "y": y}, {}, {}),
        "where_broadcast": ("Where", {"mask": np.array([[True], [False]]),
                                     "x": x[0].copy().astype(np.float16),
                                     "y": np.array(3, dtype=np.float16)}, {}, {}),
        "transpose": ("Transpose", {"x": x}, {}, {"perm": [1, 0]}),
        "transpose_int64": ("Transpose", {"x": np.arange(24, dtype=np.int64).reshape(2, 3, 4)},
                            {}, {"perm": [2, 0, 1]}),
        "transpose_bool": ("Transpose", {"x": x > 0}, {}, {"perm": [1, 0]}),
        "slice": ("Slice", {"x": x}, {"start": np.array([1], np.int64),
                  "end": np.array([3], np.int64), "axes": np.array([1], np.int64)}, {}),
        "slice_fp16_step": ("Slice", {"x": np.arange(16, dtype=np.float16).reshape(2, 8)},
                            {"start": np.array([1], np.int64), "end": np.array([8], np.int64),
                             "axes": np.array([-1], np.int64), "step": np.array([2], np.int64)}, {}),
        "expand": ("Expand", {"x": x[:1].copy()}, {"shape": np.array([2, 4, 3], np.int64)}, {}),
        "argmax_int32_ties": ("ArgMax", {"x": np.array([[1, 5, 5], [3, 3, 0]], np.int32)},
                              {}, {"axis": -1, "keepdims": 0}),
        "argmax_scalar_fp16": ("ArgMax", {"x": np.array([1, 3, 3], np.float16)},
                               {}, {"axis": 0, "keepdims": 0}),
        "reduce_l2": ("ReduceL2", {"x": x}, {}, {"axes": [-1], "keepdims": 1}),
        "reduce_l2_scalar_fp16": ("ReduceL2", {"x": x.astype(np.float16)},
                                  {}, {"keepdims": 0}),
        "conv2d": ("Conv", {"x": np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4) / 10},
                   {"weight": np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2) / 20},
                   {"pads": [1, 0, 1, 0], "strides": [2, 1]}),
        "conv2d_fp16_bias": ("Conv", {"x": np.arange(24, dtype=np.float16).reshape(1, 2, 3, 4) / 10},
                             {"weight": np.arange(16, dtype=np.float16).reshape(2, 2, 2, 2) / 20,
                              "bias": np.array([0.25, -0.25], np.float16)}, {}),
        "conv3d_stride_bias": (
            "Conv",
            {"x": np.arange(192, dtype=np.float32).reshape(2, 3, 2, 4, 4) / 50},
            {"weight": np.arange(120, dtype=np.float32).reshape(5, 3, 2, 2, 2) / 60,
             "bias": np.arange(5, dtype=np.float32) / 4},
            {"strides": [2, 2, 2]},
        ),
        "conv_transpose_stride": (
            "ConvTranspose",
            {"x": np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3) / 5},
            {"weight": np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2) / 20},
            {"strides": [2, 2], "output_padding": [1, 0]},
        ),
        "grid_sample": ("GridSample",
                        {"x": np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3),
                         "grid": np.array([[[[0.3, 0.5], [-1.2, -1.1]]]], np.float32)},
                        {}, {"mode": "linear", "padding_mode": "zeros"}),
        "roi_align": (
            "RoiAlign",
            {"x": np.arange(32, dtype=np.float32).reshape(2, 1, 4, 4),
             "boxes": np.array([[0, 0, 3, 3], [1, 0, 3, 2]], np.float32),
             "batch": np.array([1, 0], np.int64)},
            {}, {"output_height": 2, "output_width": 2, "sampling_ratio": 2,
                 "coordinate_transformation_mode": "output_half_pixel"},
        ),
        "scatter_nd_negative": (
            "ScatterND",
            {"x": x, "indices": np.array([[-1, -2], [0, 1]], np.int64),
             "updates": np.array([9, 11], np.float32)}, {}, {},
        ),
        "scatter_nd_add": (
            "ScatterND",
            {"x": x.astype(np.float16), "indices": np.array([[1], [1]], np.int64),
             "updates": np.array([[3, 2, 1], [4, 5, 6]], np.float16)},
            {}, {"reduction": "add"},
        ),
        "concat": ("Concat", {"x": x, "y": y[:, :1].copy()}, {}, {"axis": -1}),
        "concat_int64": ("Concat", {"x": np.array([[2**40], [1]], np.int64),
                                    "y": np.array([[-2**40], [2]], np.int64)}, {}, {"axis": 0}),
        "concat_empty": ("Concat", {"x": x},
                         {"empty": np.empty((2, 0), np.float32)}, {"axis": 1}),
        "layer_norm": ("LayerNormalization", {"x": x},
                       {"scale": np.array([1, 2, 3], np.float32),
                        "bias": np.array([0.5, -0.25, 0], np.float32)},
                       {"axis": -1, "epsilon": 1e-5}),
        "layer_norm_fp16": ("LayerNormalization",
                            {"x": np.arange(12, dtype=np.float16).reshape(2, 2, 3) / 3},
                            {"scale": np.ones((2, 3), np.float16)}, {"axis": -2}),
        "mul_broadcast": ("Mul", {"x": x, "y": y[0].copy()}, {}, {}),
        "mul_fp16_scalar": ("Mul", {"x": x.astype(np.float16),
                                    "y": np.array(1.702, np.float16)}, {}, {}),
        "sigmoid_fp16": ("Sigmoid", {"x": x.astype(np.float16)}, {}, {}),
        "sin_fp32": ("Sin", {"x": x / 3}, {}, {}),
        "cos_fp16": ("Cos", {"x": (x / 3).astype(np.float16)}, {}, {}),
        "erf_fp32": ("Erf", {"x": x / 3}, {}, {}),
        "sqrt_fp16": ("Sqrt", {"x": y.astype(np.float16) / 3}, {}, {}),
        "neg_fp32": ("Neg", {"x": x / 3}, {}, {}),
        "abs_fp16": ("Abs", {"x": x.astype(np.float16)}, {}, {}),
        "sign_fp32": ("Sign", {"x": x}, {}, {}),
        "log_fp32": ("Log", {"x": (y + 1) / 3}, {}, {}),
        "clip_fp16": ("Clip", {"x": x.astype(np.float16)},
                      {"min": np.array(-.5, np.float16), "max": np.array(1.5, np.float16)}, {}),
        "instance_norm_fp32": (
            "InstanceNormalization", {"x": np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4) / 5},
            {"scale": np.array([1, 2], np.float32), "bias": np.array([.5, -.5], np.float32)},
            {"epsilon": 1e-5},
        ),
        "max_fp16": ("Max", {"x": x.astype(np.float16), "minimum": np.array(1, np.float16)}, {}, {}),
        "einsum_mask": ("Einsum",
                        {"x": np.arange(12, dtype=np.float32).reshape(2, 2, 3) / 7,
                         "w": np.arange(36, dtype=np.float32).reshape(2, 3, 2, 3) / 13},
                        {}, {"equation": "bqc,bchw->bqhw"}),
        "einsum_transposed_output": (
            "Einsum", {"x": x, "w": z / 3}, {}, {"equation": "ik,kj->ji"},
        ),
        "relu_fp16": ("Relu", {"x": x.astype(np.float16)}, {}, {}),
        "greater_fp32": ("Greater", {"x": x, "y": y[0].copy()}, {}, {}),
        "greater_equal_fp32": ("GreaterOrEqual", {"x": np.array([np.nan, 0, 1, 2], np.float32),
                                                  "y": np.array(1, np.float32)}, {}, {}),
        "reduce_mean_fp32": ("ReduceMean", {"x": x / 3}, {}, {"axes": [-1], "keepdims": 0}),
        "reduce_mean_fp16": ("ReduceMean", {"x": (x / 3).astype(np.float16)}, {}, {"keepdims": 0}),
        "scatter_elements_fp16": (
            "ScatterElements",
            {"x": x.astype(np.float16), "indices": np.array([[-1, 0], [0, -2]], np.int64),
             "updates": np.array([[9, 11], [12, 13]], np.float16)}, {}, {"axis": -1},
        ),
        "pow_scalar": ("Pow", {"x": x / 3, "exponent": np.array(2, np.float32)}, {}, {}),
        "pow_fp16": ("Pow", {"x": (y / 3).astype(np.float16),
                             "exponent": np.array([1, 2, 3], np.float16)}, {}, {}),
        "split_uneven": ("Split", {"x": x},
                         {"widths": np.array([2, 1], np.int64)}, {"axis": -1}),
        "tanh_fp16": ("Tanh", {"x": x.astype(np.float16)}, {}, {}),
        "reduce_max_fp16": ("ReduceMax", {"x": x.astype(np.float16)},
                            {}, {"axes": [-1], "keepdims": 0}),
        "reduce_sum_fp32": ("ReduceSum", {"x": x},
                            {"axes": np.array([-1], np.int64)}, {"keepdims": 0}),
        "reduce_sum_int64": (
            "ReduceSum", {"x": np.array([[2**59, 2**60], [-2**59, -2**60]], np.int64)},
            {"axes": np.array([-1], np.int64)}, {"keepdims": 0},
        ),
        "resize_fp16_linear": (
            "Resize", {"x": np.arange(12, dtype=np.float16).reshape(1, 1, 3, 4) / 3},
            {"roi": np.array([0, 0, 0, 0, 1, 1, 1, 1], np.float32),
             "scales": np.array([1, 1, 2, 1.5], np.float32)},
            {"mode": "linear", "coordinate_transformation_mode": "half_pixel"},
        ),
        "resize_nearest_floor": (
            "Resize", {"x": np.arange(12, dtype=np.float32).reshape(1, 1, 3, 4)},
            {"roi": np.array([0, 0, 0, 0, 1, 1, 1, 1], np.float32),
             "scales": np.array([1, 1, 2, 2], np.float32)},
            {"mode": "nearest", "coordinate_transformation_mode": "asymmetric", "nearest_mode": "floor"},
        ),
        "resize_antialias": (
            "Resize", {"x": (np.arange(60, dtype=np.float32).reshape(1, 2, 5, 6) % 7) / 3},
            {"roi": np.array([0, 0, 1, 1], np.float32),
             "scales": np.array([.6, .5], np.float32)},
            {"mode": "linear", "coordinate_transformation_mode": "half_pixel",
             "antialias": 1, "exclude_outside": 1, "axes": [2, 3]},
        ),
        "not_bool": ("Not", {"x": x > 0}, {}, {}),
        "max_pool_fp32": ("MaxPool", {"x": -np.arange(1, 31, dtype=np.float32).reshape(1, 2, 3, 5)},
                          {}, {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [2, 2]}),
        "max_pool_fp16": ("MaxPool", {"x": -np.arange(1, 31, dtype=np.float16).reshape(1, 2, 3, 5)},
                          {}, {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [1, 1]}),
    }


def execute_case(name, runtime):
    op, inputs, const, attrs = cases()[name]

    def info(key, value):
        return helper.make_tensor_value_info(key, helper.np_dtype_to_tensor_dtype(value.dtype), value.shape)

    output_names = ["output0", "output1"] if op == "Split" else ["output"]
    node = helper.make_node(op, [*inputs, *const], output_names, **attrs)
    graph = helper.make_graph([node], name, [info(k, v) for k, v in inputs.items()],
                              [helper.make_empty_tensor_value_info(key) for key in output_names],
                              [numpy_helper.from_array(v, k) for k, v in const.items()])
    opset = 20 if op == "GridSample" else 18 if name == "resize_antialias" else 17
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=8)
    references = ReferenceEvaluator(model).run(None, inputs)
    for index, (key, expected) in enumerate(zip(output_names, references)):
        model.graph.output[index].CopyFrom(info(key, expected))
    stub = OnnxStub(model, runtime, simplify_model=False,
                    fold_static_shapes=name == "resize_antialias" or op == "Einsum")
    for key, value in inputs.items():
        stub.inputs[key].copyin_numpy(value)
    stub.run()
    for key, expected in zip(output_names, references):
        actual = stub.outputs[key].copyout_numpy()
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise AssertionError(f"Output metadata: {actual.dtype} {actual.shape}, expected {expected.dtype} {expected.shape}")
        tolerance = 1e-5 if op in {"MatMul", "Gemm"} and expected.dtype == np.float32 else 1e-3
        if np.issubdtype(expected.dtype, np.inexact):
            np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
        else:
            np.testing.assert_array_equal(actual, expected)
    return [(value.shape, str(value.dtype)) for value in references]


class TestCambriconNativeOnnx(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.runtime = backend.runtime("cambricon", 0)
        except (RuntimeError, AttributeError) as error:
            raise unittest.SkipTest(str(error)) from error

    def test_native_onnx_operators(self):
        for name in cases():
            with self.subTest(operator=name):
                execute_case(name, self.runtime)


if __name__ == "__main__":
    unittest.main()
