"""CPU Gemm transpose paths used by an exported standard linear layer."""

import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from pyinfinitensor import onnx as frontend
from pyinfinitensor.onnx import OnnxStub, backend


class TestGemmTranspose(unittest.TestCase):
    def _check(self, trans_a, trans_b):
        # Rectangular matrices distinguish row/column strides. M changes on
        # the same instance; B and the column bias remain fixed initializers.
        b = np.arange(12, dtype=np.float32).reshape(3, 4) - 3
        stored_b = b.T.copy() if trans_b else b
        bias = np.array([1, -2, 3, -4], dtype=np.float32)
        a_shape = [3, "rows"] if trans_a else ["rows", 3]
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "Gemm",
                        ["a", "b", "bias"],
                        ["y"],
                        transA=int(trans_a),
                        transB=int(trans_b),
                    )
                ],
                "transpose",
                [helper.make_tensor_value_info("a", TensorProto.FLOAT, a_shape)],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["rows", 4])],
                [
                    numpy_helper.from_array(stored_b, "b"),
                    numpy_helper.from_array(bias, "bias"),
                ],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
            ir_version=9,
        )
        # Prevent onnxsim from erasing the transpose whose kernel is tested.
        with patch.object(frontend, "simplify", side_effect=lambda m: (m, False)):
            stub = OnnxStub(model, backend.cpu_runtime())
        for rows in (2, 5, 1, 2):
            a = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3) - 2
            stored_a = a.T.copy() if trans_a else a
            stub.set_input([list(stored_a.shape)])
            stub.inputs["a"].copyin_numpy(stored_a)
            stub.run()
            np.testing.assert_array_equal(
                stub.outputs["y"].copyout_numpy(), a @ b + bias
            )

    def test_transposed_weights(self):
        self._check(False, True)

    def test_transposed_input(self):
        self._check(True, False)

    def test_both_transposed(self):
        self._check(True, True)


if __name__ == "__main__":
    unittest.main()
