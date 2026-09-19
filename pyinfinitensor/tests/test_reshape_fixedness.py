"""A Reshape output dimension follows what the target element says about it.

ONNX gives a Reshape target three kinds of element, and each says something
different about whether the dimension it describes can change:

  * a positive number names the dimension outright, so a settled element
    settles the dimension whatever the input does;
  * a zero keeps whatever the input has in that position, so the dimension
    follows that one input dimension;
  * a minus one asks for what is left once the others are taken, which is a
    function of the whole input shape.

An element that is not settled yet is a number only for the shape the graph
currently holds, and nothing can be claimed from it.
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
            "reshape_fixedness",
            [info(*x) for x in inputs],
            [info(*x) for x in outputs],
            list(initializers),
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=9,
    )
    with patch.object(frontend, "simplify", side_effect=lambda m: (m, False)):
        stub = OnnxStub(model, backend.cpu_runtime())
    # Fixedness is worked out by a graph pass rather than promised by a
    # constructor, so ask after that pass has run.
    stub.handler.shape_infer()
    return stub


def constant(name, values):
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name)


class TestReshapeFixedness(unittest.TestCase):
    def test_constant_target_settles_every_dimension_it_names(self):
        """A target of plain positive numbers settles the whole output.

        The input's batch still moves, but no output dimension follows it: each
        one is named outright by an element that cannot change.
        """
        # The placeholder batch is 1, so a target of plain numbers has to
        # account for exactly the elements that shape holds.
        stub = make_stub(
            [helper.make_node("Reshape", ["x", "target"], ["y"])],
            [("x", ["batch", 4, 6])],
            [("y", [2, 3, 4])],
            [constant("target", [2, 3, 4])],
        )

        self.assertEqual(
            [stub.outputs["y"].is_dim_dynamic(d) for d in range(3)],
            [False, False, False],
        )

    def test_a_zero_follows_the_input_dimension_it_keeps(self):
        """Zero keeps the input dimension, so it is as settled as that one is.

        Position 0 keeps a dynamic batch and position 1 keeps a fixed 4, which
        is the whole distinction: both are zeros in the target, and they must
        not come out the same.
        """
        stub = make_stub(
            [helper.make_node("Reshape", ["x", "target"], ["y"])],
            [("x", ["batch", 4, 6])],
            [("y", ["batch", 4, 6])],
            [constant("target", [0, 0, 6])],
        )

        self.assertEqual(
            [stub.outputs["y"].is_dim_dynamic(d) for d in range(3)],
            [True, False, False],
        )

    def test_an_inferred_dimension_follows_the_whole_input(self):
        """Minus one is what is left over, so any moving dimension moves it.

        The named 6 beside it stays settled, which is what says the rule is
        per-element rather than a verdict on the whole output.
        """
        stub = make_stub(
            [helper.make_node("Reshape", ["x", "target"], ["y"])],
            [("x", ["batch", 4, 6])],
            [("y", ["batch_times_4", 6])],
            [constant("target", [-1, 6])],
        )

        self.assertEqual(
            [stub.outputs["y"].is_dim_dynamic(d) for d in range(2)],
            [True, False],
        )

    def test_an_inferred_dimension_is_settled_when_the_input_is(self):
        """Nothing is left to move, so the leftover cannot move either.

        A shape a model spells out in full is still one the graph may be handed
        again, so saying the dimensions are fixed takes pinning them: an
        undeclared dimension stays replaceable by design. Once pinned, every
        dimension the leftover is worked out from is settled, and so is it.
        """
        stub = make_stub(
            [helper.make_node("Reshape", ["x", "target"], ["y"])],
            [("x", [2, 4, 6])],
            [("y", [8, 6])],
            [constant("target", [-1, 6])],
        )
        stub.pin_dims("x", [0, 1, 2])
        stub.handler.shape_infer()

        self.assertEqual(
            [stub.outputs["y"].is_dim_dynamic(d) for d in range(2)],
            [False, False],
        )

    def test_a_moving_target_element_settles_nothing(self):
        """An element read off the input's own shape is this shape's number only.

        `batch` reaches the target through Shape/Gather/Unsqueeze/Concat, so the
        first element moves and the 32 beside it does not.
        """
        stub = make_stub(
            [
                helper.make_node("Shape", ["x"], ["shape"]),
                helper.make_node("Gather", ["shape", "zero"], ["batch"], axis=0),
                helper.make_node("Unsqueeze", ["batch", "zero"], ["batch_1d"]),
                helper.make_node("Concat", ["batch_1d", "tail"], ["target"], axis=0),
                helper.make_node("Reshape", ["x", "target"], ["y"]),
            ],
            [("x", ["batch", 4, 8])],
            [("y", ["batch", 32])],
            [constant("zero", 0), constant("tail", [32])],
        )

        self.assertEqual(
            [stub.outputs["y"].is_dim_dynamic(d) for d in range(2)],
            [True, False],
        )

    def test_mixed_target_kinds_are_judged_one_element_at_a_time(self):
        """One target holding all three kinds at once.

        Position 0 keeps a dynamic batch, position 1 is named outright, and
        position 2 is left over -- which the moving batch reaches, because the
        number of elements does.
        """
        stub = make_stub(
            [helper.make_node("Reshape", ["x", "target"], ["y"])],
            [("x", ["batch", 4, 6])],
            [("y", ["batch", 3, "rest"])],
            [constant("target", [0, 3, -1])],
        )

        self.assertEqual(
            [stub.outputs["y"].is_dim_dynamic(d) for d in range(3)],
            [True, False, True],
        )

    def test_a_settled_width_reaches_the_shape_subgraph(self):
        """The point of the rule: a settled dimension lets what reads it fold.

        Reading the width back off the output leaves a chain that only follows
        settled dimensions, so the fold can take it. Without the rule the width
        is called dynamic and the chain stays.
        """
        stub = make_stub(
            [
                helper.make_node("Reshape", ["x", "target"], ["y"]),
                helper.make_node("Shape", ["y"], ["out_shape"]),
                helper.make_node("Gather", ["out_shape", "one"], ["width"], axis=0),
                helper.make_node("Add", ["width", "zero"], ["width_out"]),
            ],
            [("x", ["batch", 4, 6])],
            [("y", ["batch", 24]), ("width_out", [], TensorProto.INT64)],
            [constant("target", [0, 24]), constant("one", 1), constant("zero", 0)],
        )

        self.assertFalse(stub.outputs["y"].is_dim_dynamic(1))
        self.assertTrue(stub.tensors["width"].is_shape_value_wholly_fixed())
        dropped = stub.fold_shape_subgraph()
        self.assertGreater(dropped, 0)
        self.assertEqual(stub.fold_shape_subgraph(), 0)

        # The rule only claims what cannot change, so the graph still has to
        # give the right answer under every shape it is handed.
        for batch in (2, 1, 5, 2):
            x = np.arange(batch * 24, dtype=np.float32).reshape(batch, 4, 6)
            stub.set_input([list(x.shape)])
            stub.inputs["x"].copyin_numpy(x)
            stub.run()
            np.testing.assert_array_equal(
                stub.outputs["y"].copyout_numpy().reshape(batch, 24),
                x.reshape(batch, 24),
            )
            self.assertEqual(stub.outputs["width_out"].copyout_int64(), [24])

    def test_values_stay_right_across_shapes_for_every_target_kind(self):
        # A target naming every dimension outright fits one input size only, so
        # it is asked of a static input; the other two follow the batch.
        for name, dims, target, expected in (
            ("zero_and_named", ["batch", 4, 6], [0, 24], lambda b: (b, 24)),
            ("inferred", ["batch", 4, 6], [-1, 6], lambda b: (b * 4, 6)),
            ("all_named", [2, 4, 6], [8, 6], lambda b: (8, 6)),
        ):
            with self.subTest(target=name):
                shapes = (2, 1, 5, 2) if name != "all_named" else (2,)
                stub = make_stub(
                    [helper.make_node("Reshape", ["x", "target"], ["y"])],
                    [("x", dims)],
                    [("y", ["a", "b"] if len(target) == 2 else ["a", "b", "c"])],
                    [constant("target", target)],
                )
                for batch in shapes:
                    x = np.arange(batch * 24, dtype=np.float32).reshape(batch, 4, 6)
                    stub.set_input([list(x.shape)])
                    stub.inputs["x"].copyin_numpy(x)
                    stub.run()
                    want = x.reshape(*expected(batch))
                    np.testing.assert_array_equal(
                        stub.outputs["y"].copyout_numpy().reshape(want.shape), want
                    )


if __name__ == "__main__":
    unittest.main()
