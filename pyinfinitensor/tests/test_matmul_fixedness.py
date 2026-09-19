"""MatMul shape dependencies exclude reduction dimensions and broadcast bias."""

import unittest
from unittest.mock import patch

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from pyinfinitensor import onnx as frontend
from pyinfinitensor.onnx import OnnxStub, backend


def make_stub(nodes, inputs, outputs, initializers=()):
    def info(name, dims, dtype=TensorProto.FLOAT):
        return helper.make_tensor_value_info(name, dtype, dims)

    model = helper.make_model(
        helper.make_graph(
            nodes,
            "matmul_fixedness",
            [info(*x) for x in inputs],
            [info(*x) for x in outputs],
            list(initializers),
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=9,
    )
    with patch.object(frontend, "simplify", side_effect=lambda m: (m, False)):
        stub = OnnxStub(model, backend.cpu_runtime())
    # Fixedness propagation is a graph inference pass, not a constructor
    # guarantee. Exercise the rule after that pass, including initial size 1.
    stub.handler.shape_infer()
    return stub


class TestMatmulFixedness(unittest.TestCase):
    def test_fixed_width_folds_after_biased_projection(self):
        weights = np.arange(12, dtype=np.float32).reshape(3, 4)
        bias = np.array([1, 2, 3, 4], dtype=np.float32)
        stub = make_stub(
            [
                helper.make_node("MatMul", ["x", "w"], ["projection"]),
                helper.make_node("Add", ["projection", "bias"], ["y"]),
                helper.make_node("Shape", ["y"], ["shape"]),
                helper.make_node("Gather", ["shape", "axis"], ["width"]),
                helper.make_node("Add", ["width", "zero"], ["width_out"]),
            ],
            [("x", ["batch", "seq", 3])],
            [("y", ["batch", "seq", 4]), ("width_out", [], TensorProto.INT64)],
            [
                numpy_helper.from_array(a, n)
                for n, a in (
                    ("w", weights),
                    ("bias", bias),
                    ("axis", np.array(2, dtype=np.int64)),
                    ("zero", np.array(0, dtype=np.int64)),
                )
            ],
        )
        self.assertFalse(stub.outputs["y"].is_dim_dynamic(2))
        self.assertTrue(stub.tensors["width"].is_shape_value_wholly_fixed())
        self.assertEqual(stub.fold_shape_subgraph(), 2)
        self.assertEqual(stub.fold_shape_subgraph(), 0)
        for batch, seq in ((2, 5), (1, 1), (3, 7), (2, 5)):
            x = np.arange(batch * seq * 3, dtype=np.float32).reshape(batch, seq, 3)
            stub.set_input([list(x.shape)])
            stub.inputs["x"].copyin_numpy(x)
            stub.run()
            np.testing.assert_array_equal(
                stub.outputs["y"].copyout_numpy(), x @ weights + bias
            )
            self.assertEqual(stub.outputs["width_out"].copyout_numpy().item(), 4)

    def test_transpose_and_dynamic_reduction(self):
        for ta, tb in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(trans_a=ta, trans_b=tb):
                ashape = ["k", "m"] if ta else ["m", "k"]
                bshape = [4, "k"] if tb else ["k", 4]
                stub = make_stub(
                    [
                        helper.make_node(
                            "Gemm", ["a", "b"], ["y"], transA=int(ta), transB=int(tb)
                        )
                    ],
                    [("a", ashape), ("b", bshape)],
                    [("y", ["m", 4])],
                )
                self.assertTrue(stub.outputs["y"].is_dim_dynamic(0))
                self.assertFalse(stub.outputs["y"].is_dim_dynamic(1))
                for m, k in ((2, 3), (5, 2), (1, 5), (2, 3)):
                    a = np.arange(m * k, dtype=np.float32).reshape(m, k)
                    b = np.arange(k * 4, dtype=np.float32).reshape(k, 4)
                    sa, sb = a.T.copy() if ta else a, b.T.copy() if tb else b
                    stub.set_input([list(sa.shape), list(sb.shape)])
                    stub.inputs["a"].copyin_numpy(sa)
                    stub.inputs["b"].copyin_numpy(sb)
                    stub.run()
                    np.testing.assert_array_equal(
                        stub.outputs["y"].copyout_numpy(), a @ b
                    )
                    self.assertFalse(stub.outputs["y"].is_dim_dynamic(1))

    def test_broadcast_axis_currently_one_remains_dynamic(self):
        for batched_a in (True, False):
            with self.subTest(batched_a=batched_a):
                ashape = ["batch", 2, 3] if batched_a else [2, 3]
                bshape = [3, 4] if batched_a else ["batch", 3, 4]
                stub = make_stub(
                    [helper.make_node("MatMul", ["a", "b"], ["y"])],
                    [("a", ashape), ("b", bshape)],
                    [("y", ["batch", 2, 4])],
                )
                # Fully static ONNX inputs keep the frontend's legacy mutable
                # shape contract unless explicitly pinned. Only the mixed
                # symbolic input declares its non-batch axes fixed by itself.
                stub.pin_dims("b" if batched_a else "a", [0, 1])
                self.assertEqual(
                    [stub.outputs["y"].is_dim_dynamic(i) for i in range(3)],
                    [True, False, False],
                )
                for batch in (1, 3, 2, 1):
                    a = np.arange(
                        (batch if batched_a else 1) * 6, dtype=np.float32
                    ).reshape((batch, 2, 3) if batched_a else (2, 3))
                    b = np.arange(
                        (1 if batched_a else batch) * 12, dtype=np.float32
                    ).reshape((3, 4) if batched_a else (batch, 3, 4))
                    stub.set_input([list(a.shape), list(b.shape)])
                    stub.inputs["a"].copyin_numpy(a)
                    stub.inputs["b"].copyin_numpy(b)
                    stub.run()
                    np.testing.assert_array_equal(
                        stub.outputs["y"].copyout_numpy(), a @ b
                    )


if __name__ == "__main__":
    unittest.main()
