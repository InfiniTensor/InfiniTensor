"""Run explicitly with an InfiniOps SDK containing RmsNorm/Cat/Softmax/Abs/Erf.

Missing capabilities fail this suite; they are never skipped.
"""
import json
import math
import unittest

import numpy as np
from onnx import TensorProto
from pyinfinitensor import backend


class TestInfiniOpsBindings(unittest.TestCase):
    def handler(self):
        return backend.GraphHandler(backend.InfiniRuntime(backend.default_infini_device(), 0))

    def test_new_sdk_operators_and_schema(self):
        schemas = json.loads(backend.infini_operator_schemas())
        for name in ("Abs", "Erf"):
            self.assertIn(name, schemas)
            handler = self.handler()
            x = handler.tensor([2, 3], TensorProto.FLOAT)
            y = handler.tensor([2, 3], TensorProto.FLOAT)
            x.set_input()
            y.set_output()
            op = handler.infini_op(name, {"input": [x]}, {"out": [y]})
            self.assertIsInstance(op, backend.InfiniOp)
            handler.data_malloc()
            values = np.array([-2, -1, 0, .5, 1, 2], np.float32).reshape(2, 3)
            x.copyin_numpy(values)
            handler.run()
            reference = np.abs(values) if name == "Abs" else np.vectorize(math.erf)(values)
            np.testing.assert_allclose(y.copyout_numpy(), reference, atol=1e-6)

    def test_single_and_multiple_tensor_lists(self):
        for count in (1, 2):
            handler = self.handler()
            inputs = [handler.tensor([1, 3], TensorProto.FLOAT) for _ in range(count)]
            y = handler.tensor([count, 3], TensorProto.FLOAT)
            for tensor in inputs:
                tensor.set_input()
            y.set_output()
            handler.infini_op("Cat", {"tensors": inputs}, {"out": [y]}, {"dim": 0})
            handler.data_malloc()
            for i, tensor in enumerate(inputs):
                tensor.copyin_float([i, i + 1, i + 2])
            handler.run()
            np.testing.assert_array_equal(y.copyout_numpy(), [[i, i + 1, i + 2] for i in range(count)])

    def test_optional_dtype_accepts_onnx_code_and_canonical_name(self):
        for dtype in (TensorProto.DOUBLE, "float64"):
            handler = self.handler()
            x = handler.tensor([1, 3], TensorProto.FLOAT)
            y = handler.tensor([1, 3], TensorProto.DOUBLE)
            x.set_input()
            y.set_output()
            handler.infini_op("Softmax", {"input": [x]}, {"out": [y]}, {"dim": -1, "dtype": dtype})
            handler.data_malloc()
            x.copyin_float([1, 2, 3])
            handler.run()
            expected = np.exp(np.array([[1, 2, 3]], np.float64))
            expected /= expected.sum()
            np.testing.assert_allclose(y.copyout_numpy(), expected, atol=1e-12)

    def test_explicit_dynamic_shapes_with_intermediate_input(self):
        handler = self.handler()
        x = handler.tensor([1, 4], TensorProto.FLOAT)
        w = handler.tensor([4], TensorProto.FLOAT)
        intermediate = handler.identity(x, None)
        y = handler.tensor([1, 4], TensorProto.FLOAT)
        x.set_input()
        w.set_weight()
        y.set_output()
        op = handler.infini_op("RmsNorm", {"input": [intermediate], "weight": [w]}, {"out": [y]}, {"eps": 1e-6})
        for batch in (1, 3, 1):
            handler.change_shape([batch, 4], x.fuid())
            op.set_output_shapes([[batch, 4]])
            handler.shape_infer()
            handler.data_malloc()
            x.copyin_float([1, 2, 3, 4] * batch)
            w.copyin_float([1] * 4)
            handler.run()
            np.testing.assert_allclose(y.copyout_numpy(), np.tile(np.arange(1, 5) / np.sqrt(7.5 + 1e-6), (batch, 1)), atol=1e-6)

    def test_legacy_unary_semantics(self):
        values = np.array([[-8, -2, -.5, 0, .5, 2]], np.float32)
        for name, dtype in (("hardSigmoid", TensorProto.FLOAT), ("hardSigmoid", TensorProto.DOUBLE), ("silu", TensorProto.FLOAT), ("gelu", TensorProto.FLOAT)):
            values = values.astype(np.float64 if dtype == TensorProto.DOUBLE else np.float32)
            handler = self.handler()
            x = handler.tensor([1, 6], dtype)
            x.set_input()
            if name == "hardSigmoid":
                y = handler.hardSigmoid(x, None, .3, .1)
                expected = np.clip(values * .3 + .1, 0, 1)
            else:
                y = getattr(handler, name)(x, None)
                expected = (values / (1 + np.exp(-values)) if name == "silu"
                            else values * .5 * (1 + np.vectorize(math.erf)(values / np.sqrt(2))))
            y.set_output()
            handler.data_malloc()
            x.copyin_numpy(values)
            handler.run()
            np.testing.assert_allclose(y.copyout_numpy(), expected, rtol=1e-12 if dtype == TensorProto.DOUBLE else 1e-5, atol=1e-12 if dtype == TensorProto.DOUBLE else 1e-6)

    def test_matmul_preserves_batch_axes_transpose_and_bias(self):
        handler = self.handler()
        a = handler.tensor([2, 3, 2, 4], TensorProto.FLOAT)
        b = handler.tensor([1, 3, 5, 4], TensorProto.FLOAT)
        bias = handler.tensor([5], TensorProto.FLOAT)
        for tensor in (a, b, bias):
            tensor.set_input()
        y = handler.matmul(a, b, None, False, True, bias, backend.ActType.Linear, "default")
        y.set_output()
        handler.data_malloc()
        av = np.arange(48, dtype=np.float32).reshape(2, 3, 2, 4) / 10
        bv = np.arange(60, dtype=np.float32).reshape(1, 3, 5, 4) / 20
        cv = np.arange(5, dtype=np.float32)
        a.copyin_numpy(av)
        b.copyin_numpy(bv)
        bias.copyin_numpy(cv)
        handler.run()
        np.testing.assert_allclose(y.copyout_numpy(), av @ bv.swapaxes(-1, -2) + cv, rtol=1e-5, atol=1e-5)

    def test_invalid_names_types_and_overflow_do_not_execute(self):
        for attributes in ({"dim": True}, {"dim": 2**64 - 1}, {"dim": [2**64 - 1]}, {"dim": 0, "extra": 1}):
            handler = self.handler()
            x = handler.tensor([1, 3], TensorProto.FLOAT)
            y = handler.tensor([1, 3], TensorProto.FLOAT)
            x.set_input()
            y.set_output()
            handler.infini_op("Cat", {"tensors": [x]}, {"out": [y]}, attributes)
            handler.data_malloc()
            x.copyin_float([1, 2, 3])
            y.copyin_float([77] * 3)
            with self.assertRaises(RuntimeError):
                handler.run()
            np.testing.assert_array_equal(y.copyout_numpy(), [[77] * 3])


if __name__ == "__main__":
    unittest.main()
