"""Two shape operators doing the same work are kept as one.

An exporter commonly reads the same dimension off the same tensor several times
over, once for each place that needs it, so a graph arrives holding several
operators of the same kind reading the same tensors with the same attributes.
Each of them computes what the others already did.

This is not what `fold_shape_subgraph` does. A fold replaces a settled
computation with the constant it works out to, and leaves alone anything that
still follows the input. Two operators that both still follow the input are
untouched by it however alike they are, which is exactly the case here.
"""

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
            "shape_cse",
            [info(*x) for x in inputs],
            [info(*x) for x in outputs],
            list(initializers),
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=9,
    )
    with patch.object(frontend, "simplify", side_effect=lambda m: (m, False)):
        stub = OnnxStub(model, backend.cpu_runtime())
    stub.handler.shape_infer()
    return stub


def constant(name, values):
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name)


class TestShapeCommonSubexpressions(unittest.TestCase):
    def test_two_identical_shape_reads_become_one(self):
        """The same dimension read off the same tensor twice.

        Both `Shape` operators read `x` and both `Gather` operators take element
        0 of what they read, so the second of each pair does what the first
        already did.
        """
        stub = make_stub(
            [
                helper.make_node("Shape", ["x"], ["s1"]),
                helper.make_node("Shape", ["x"], ["s2"]),
                helper.make_node("Gather", ["s1", "zero"], ["b1"], axis=0),
                helper.make_node("Gather", ["s2", "zero"], ["b2"], axis=0),
                helper.make_node("Unsqueeze", ["b1", "zero"], ["u1"]),
                helper.make_node("Unsqueeze", ["b2", "zero"], ["u2"]),
                helper.make_node("Concat", ["u1", "u2"], ["pair"], axis=0),
            ],
            [("x", ["batch", 4])],
            [("pair", [2], TensorProto.INT64)],
            [constant("zero", 0)],
        )

        before = stub.handler.operator_count()
        removed = stub.merge_duplicate_shape_operators()

        # One of each duplicated pair goes: Shape, Gather, Unsqueeze.
        self.assertEqual(removed, 3)
        self.assertEqual(stub.handler.operator_count(), before - 3)

        # A second pass has nothing left to do.
        self.assertEqual(stub.merge_duplicate_shape_operators(), 0)

        # And the graph still answers with both copies of the dimension.
        for batch in (2, 1, 7, 2):
            stub.set_input([[batch, 4]])
            stub.run()
            self.assertEqual(
                stub.outputs["pair"].copyout_int64(),
                [batch, batch],
                "batch {}".format(batch),
            )

    def test_different_attributes_are_left_alone(self):
        """Same kind, same input, different axis: not the same work."""
        stub = make_stub(
            [
                helper.make_node("Shape", ["x"], ["s"]),
                helper.make_node("Gather", ["s", "zero"], ["g0"], axis=0),
                helper.make_node("Gather", ["s", "one"], ["g1"], axis=0),
                helper.make_node("Unsqueeze", ["g0", "zero"], ["u0"]),
                helper.make_node("Unsqueeze", ["g1", "zero"], ["u1"]),
                helper.make_node("Concat", ["u0", "u1"], ["both"], axis=0),
            ],
            [("x", ["batch", 4])],
            [("both", [2], TensorProto.INT64)],
            [constant("zero", 0), constant("one", 1)],
        )

        before = stub.handler.operator_count()
        # The two Gathers read different elements, so neither Gather nor the
        # Unsqueeze that follows it is a copy of the other.
        self.assertEqual(stub.merge_duplicate_shape_operators(), 0)
        self.assertEqual(stub.handler.operator_count(), before)

        stub.set_input([[5, 4]])
        stub.run()
        self.assertEqual(stub.outputs["both"].copyout_int64(), [5, 4])

    def test_different_inputs_are_left_alone(self):
        """Same kind and attributes, different tensors: different work.

        The two inputs happen to hold the same shape here, which is what makes
        this worth stating: matching is on the tensors themselves, not on the
        numbers they currently carry.
        """
        stub = make_stub(
            [
                helper.make_node("Shape", ["a"], ["sa"]),
                helper.make_node("Shape", ["b"], ["sb"]),
                helper.make_node("Concat", ["sa", "sb"], ["both"], axis=0),
            ],
            [("a", ["batch", 4]), ("b", ["batch", 4])],
            [("both", [4], TensorProto.INT64)],
        )

        self.assertEqual(stub.merge_duplicate_shape_operators(), 0)

        stub.set_input([[3, 4], [3, 4]])
        stub.run()
        self.assertEqual(stub.outputs["both"].copyout_int64(), [3, 4, 3, 4])

    def test_a_graph_output_keeps_its_producer(self):
        """A duplicate whose result the graph was asked for stays.

        The graph promises every tensor it holds is either produced by an
        operator or given from outside, so an output cannot lose its producer
        even when another operator computes the same thing.
        """
        stub = make_stub(
            [
                helper.make_node("Shape", ["x"], ["s1"]),
                helper.make_node("Shape", ["x"], ["s2"]),
                helper.make_node("Concat", ["s1"], ["joined"], axis=0),
            ],
            [("x", ["batch", 4])],
            [("s2", [2], TensorProto.INT64), ("joined", [2], TensorProto.INT64)],
        )

        # `s2` is asked for by name, so the `Shape` producing it has to stay
        # whichever of the two is treated as the copy.
        stub.merge_duplicate_shape_operators()
        stub.set_input([[6, 4]])
        stub.run()
        self.assertEqual(stub.outputs["s2"].copyout_int64(), [6, 4])
        self.assertEqual(stub.outputs["joined"].copyout_int64(), [6, 4])

    def test_data_operators_are_not_merged(self):
        """Arithmetic on data is left alone even when it looks duplicated.

        Two `Add`s over the same activations are the same computation, but this
        pass is about the shape subgraph: the tensors here carry data, not
        dimensions, and merging them is a different question with different
        risks. Saying so keeps the pass to what it can argue for.
        """
        stub = make_stub(
            [
                helper.make_node("Add", ["x", "x"], ["d1"]),
                helper.make_node("Add", ["x", "x"], ["d2"]),
                helper.make_node("Mul", ["d1", "d2"], ["y"]),
            ],
            [("x", ["batch", 4])],
            [("y", ["batch", 4])],
        )

        before = stub.handler.operator_count()
        self.assertEqual(stub.merge_duplicate_shape_operators(), 0)
        self.assertEqual(stub.handler.operator_count(), before)

        x = np.arange(8, dtype=np.float32).reshape(2, 4)
        stub.set_input([[2, 4]])
        stub.inputs["x"].copyin_numpy(x)
        stub.run()
        np.testing.assert_array_equal(
            stub.outputs["y"].copyout_numpy().reshape(2, 4), (x + x) * (x + x)
        )

    def test_merging_survives_a_reshape_that_reads_the_result(self):
        """The merged result still drives a Reshape across shapes.

        A target is built twice over from the same dimension, which is the shape
        of a real export, and the reshape has to keep working once the copies
        are down to one.
        """
        stub = make_stub(
            [
                helper.make_node("Shape", ["x"], ["s1"]),
                helper.make_node("Shape", ["x"], ["s2"]),
                helper.make_node("Gather", ["s1", "zero"], ["b1"], axis=0),
                helper.make_node("Gather", ["s2", "zero"], ["b2"], axis=0),
                helper.make_node("Unsqueeze", ["b1", "zero"], ["u1"]),
                helper.make_node("Unsqueeze", ["b2", "zero"], ["u2"]),
                helper.make_node("Concat", ["u1", "tail"], ["target"], axis=0),
                helper.make_node("Concat", ["u2", "tail"], ["target2"], axis=0),
                helper.make_node("Reshape", ["x", "target"], ["y"]),
                helper.make_node("Reshape", ["x", "target2"], ["y2"]),
            ],
            [("x", ["batch", 3, 4])],
            [("y", ["batch", 12]), ("y2", ["batch", 12])],
            [constant("zero", 0), constant("tail", [12])],
        )

        removed = stub.merge_duplicate_shape_operators()
        self.assertGreater(removed, 0)

        for batch in (2, 1, 5, 2):
            x = np.arange(batch * 12, dtype=np.float32).reshape(batch, 3, 4)
            stub.set_input([list(x.shape)])
            stub.inputs["x"].copyin_numpy(x)
            stub.run()
            want = x.reshape(batch, 12)
            np.testing.assert_array_equal(
                stub.outputs["y"].copyout_numpy().reshape(want.shape), want
            )
            np.testing.assert_array_equal(
                stub.outputs["y2"].copyout_numpy().reshape(want.shape), want
            )

    def test_optimize_merges_as_well_as_folds(self):
        """`optimize` is where a caller expects both to have happened."""
        stub = make_stub(
            [
                helper.make_node("Shape", ["x"], ["s1"]),
                helper.make_node("Shape", ["x"], ["s2"]),
                helper.make_node("Gather", ["s1", "zero"], ["b1"], axis=0),
                helper.make_node("Gather", ["s2", "zero"], ["b2"], axis=0),
                helper.make_node("Unsqueeze", ["b1", "zero"], ["u1"]),
                helper.make_node("Unsqueeze", ["b2", "zero"], ["u2"]),
                helper.make_node("Concat", ["u1", "u2"], ["pair"], axis=0),
            ],
            [("x", ["batch", 4])],
            [("pair", [2], TensorProto.INT64)],
            [constant("zero", 0)],
        )

        before = stub.handler.operator_count()
        stub.optimize()
        self.assertLess(stub.handler.operator_count(), before)

        stub.set_input([[4, 4]])
        stub.run()
        self.assertEqual(stub.outputs["pair"].copyout_int64(), [4, 4])


if __name__ == "__main__":
    unittest.main()
