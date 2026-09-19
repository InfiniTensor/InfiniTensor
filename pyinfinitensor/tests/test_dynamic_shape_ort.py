"""Check a dynamic-shape model against onnxruntime, shape by shape.

A model whose target shapes are computed at run time can be wrong in a way no
single inference reveals: the first shape may be right while the second reads a
buffer laid out for the first, or a shape smaller than one already seen may
leave the tail of the earlier result in place. So each model is given a series
of shapes through one instance, never reloaded, and every output is compared
against onnxruntime given the same input.

The tolerance is the one the project asks for, `rtol=1e-4` and `atol=1e-5`. Both
sides evaluate the same graph in float32 on the CPU, but they do not evaluate it
the same way -- the order a sum is accumulated in is a kernel's own business --
so the results agree to within rounding rather than exactly. The observed
disagreement is some four orders of magnitude below this bound, which leaves it
loose enough not to be brittle and tight enough that a real defect cannot hide
under it.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from pyinfinitensor import backend
from pyinfinitensor import onnx as onnx_frontend
from pyinfinitensor.onnx import OnnxStub

# The demo's models rather than copies of them: they are the models the project
# demonstrates, so testing anything else would leave those untested.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "python"))

from dynamic_shape_inference import (  # noqa: E402
    batch_model,
    image_model,
    sequence_model,
    shape_of,
)

RTOL = 1e-4
ATOL = 1e-5

# Sizes picked for what each transition exercises rather than for variety: a
# repeat takes the path where nothing needs laying out again, a rise needs more
# room than the last shape had, and a fall must not read what the larger one
# left behind. The batch series is the one the project names, 1, 2, 8, 3, 1.
BATCH_SIZES = [1, 2, 8, 3, 1]
IMAGE_SIZES = [
    {"batch": 1, "height": 8, "width": 8},
    {"batch": 1, "height": 8, "width": 8},
    {"batch": 2, "height": 12, "width": 8},
    {"batch": 1, "height": 8, "width": 12},
    {"batch": 1, "height": 4, "width": 4},
]
# Two dimensions moving at once, and independently: a length that grows while
# the batch shrinks separates a shape laid out for one of them from a shape
# laid out for the other.
SEQUENCE_SIZES = [(1, 1), (2, 7), (8, 3), (3, 16), (1, 1)]


def onnxruntime_or_skip():
    try:
        from onnxruntime import InferenceSession
    except ImportError:  # pragma: no cover - depends on the environment
        raise unittest.SkipTest("onnxruntime is not installed")
    return InferenceSession


def slice_model(start, end, computed=False):
    end_name = "raw_ends" if computed else "ends"
    nodes = (
        [helper.make_node("Identity", [end_name], ["ends"])] if computed else []
    )
    nodes.append(helper.make_node("Slice", ["x", "starts", "ends", "axes"], ["y"]))
    return helper.make_model(
        helper.make_graph(
            nodes,
            "slice_bounds",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 5])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", None])],
            [
                numpy_helper.from_array(np.array([start], dtype=np.int64), "starts"),
                numpy_helper.from_array(np.array([end], dtype=np.int64), end_name),
                numpy_helper.from_array(np.array([1], dtype=np.int64), "axes"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=8,
    )


class TestAgainstOnnxRuntime(unittest.TestCase):
    def compare(self, model, schedule, simplify):
        """Run every shape in `schedule` through one instance and compare.

        `simplify` says whether the frontend's simplification pass runs, which
        is worth both ways: with it is what a caller gets by default, and
        without it the shape subgraph is certainly still in the graph to be
        computed at run time rather than having been folded away beforehand.
        """
        InferenceSession = onnxruntime_or_skip()
        session = InferenceSession(model.SerializeToString())
        inputs = list(model.graph.input)

        if simplify:
            stub = OnnxStub(model, backend.cpu_runtime())
        else:
            with patch.object(
                onnx_frontend, "simplify", side_effect=lambda c: (c, False)
            ):
                stub = OnnxStub(model, backend.cpu_runtime())

        rng = np.random.default_rng(0)
        worst = 0.0
        for sizes in schedule:
            fallback = next(iter(sizes.values()))
            shapes = [shape_of(i, sizes, fallback) for i in inputs]
            stub.set_input(shapes)

            feeds = {}
            for value_info, shape in zip(inputs, shapes):
                data = rng.standard_normal(shape).astype(np.float32)
                stub.tensors[value_info.name].copyin_numpy(data)
                feeds[value_info.name] = data
            stub.run()

            name, output = next(iter(stub.outputs.items()))
            want = session.run(None, feeds)[0]

            # The shape is checked before the values, because comparing values
            # through a reshape would hide a wrong shape of the right size.
            self.assertEqual(
                tuple(output.shape()),
                tuple(want.shape),
                "output {} has the wrong shape at {}".format(name, sizes),
            )
            got = np.asarray(output.copyout_float(), dtype=np.float32).reshape(
                want.shape
            )
            np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)
            if got.size:
                worst = max(worst, float(np.abs(got - want).max()))
        return worst

    def test_dynamic_batch_matches_onnxruntime(self):
        self.compare(
            batch_model(), [{"batch": n} for n in BATCH_SIZES], simplify=False
        )

    def test_dynamic_batch_matches_onnxruntime_by_default(self):
        self.compare(
            batch_model(), [{"batch": n} for n in BATCH_SIZES], simplify=True
        )

    def test_dynamic_image_matches_onnxruntime(self):
        self.compare(image_model(), IMAGE_SIZES, simplify=False)

    def test_dynamic_image_matches_onnxruntime_by_default(self):
        self.compare(image_model(), IMAGE_SIZES, simplify=True)

    def test_negative_gather_indices_survive_shape_updates(self):
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Gather", ["x", "indices"], ["y"], axis=1)],
                "negative_gather",
                [
                    helper.make_tensor_value_info(
                        "x", TensorProto.FLOAT, ["batch", 3]
                    )
                ],
                [
                    helper.make_tensor_value_info(
                        "y", TensorProto.FLOAT, ["batch", 2]
                    )
                ],
                [
                    numpy_helper.from_array(
                        np.array([-1, -3], dtype=np.int64), "indices"
                    )
                ],
            ),
            opset_imports=[helper.make_opsetid("", 13)],
            ir_version=8,
        )
        self.compare(model, [{"batch": n} for n in BATCH_SIZES], simplify=False)

    def test_slice_int64_sentinels_with_static_and_computed_bounds(self):
        limits = np.iinfo(np.int64)
        for computed in (False, True):
            with self.subTest(computed=computed):
                self.compare(
                    slice_model(limits.min, limits.max, computed),
                    [{"batch": n} for n in BATCH_SIZES],
                    simplify=False,
                )

    def test_empty_slice_stays_empty_from_first_import(self):
        self.compare(
            slice_model(2, 2),
            [{"batch": n} for n in BATCH_SIZES],
            simplify=False,
        )

    def test_agreement_is_far_inside_the_bound(self):
        """The bound is not doing the work; the results genuinely agree.

        A tolerance passes either because the two sides agree or because it was
        set wide enough to cover a difference that matters. Recording how far
        inside it the worst disagreement actually falls says which.
        """
        worst = self.compare(
            batch_model(), [{"batch": n} for n in BATCH_SIZES], simplify=False
        )
        self.assertLess(worst, ATOL / 100)

    def _serve(self, stub, inputs, sizes):
        """Run one inference at `sizes`, and return what came out and went in.

        The input depends on the shape and on nothing else, so the same shape
        can be served again later and given the same data to work on.
        """
        fallback = next(iter(sizes.values()))
        shapes = [shape_of(i, sizes, fallback) for i in inputs]
        stub.set_input(shapes)
        feeds = {}
        for value_info, shape in zip(inputs, shapes):
            rng = np.random.default_rng([len(shape), *shape])
            data = rng.standard_normal(shape).astype(np.float32)
            stub.tensors[value_info.name].copyin_numpy(data)
            feeds[value_info.name] = data
        stub.run()
        output = next(iter(stub.outputs.values()))
        return output, feeds

    def test_folding_the_shape_subgraph_changes_no_result(self):
        """Folding is an optimisation, so it must not move a single value.

        The fold replaces what is settled under every shape the graph may be
        given. On an ordinary import that is nothing: the frontend simplifies
        what it loads, and the simplifier has already folded whatever follows
        from the declared shapes. What it has not seen is the deployment, so a
        dimension pinned here is what gives the fold something to remove.

        A whole series is served first, and the shape the pinning settles on is
        then served again. Its result must be identical rather than close: an
        optimisation that moves an answer at all has moved it.
        """
        InferenceSession = onnxruntime_or_skip()
        model = sequence_model()
        session = InferenceSession(model.SerializeToString())
        inputs = list(model.graph.input)
        stub = OnnxStub(model, backend.cpu_runtime())

        series = [{"batch": b, "length": t} for b, t in SEQUENCE_SIZES]
        for sizes in series:
            output, feeds = self._serve(stub, inputs, sizes)
            want = session.run(None, feeds)[0]
            self.assertEqual(tuple(output.shape()), want.shape)
            got = np.asarray(output.copyout_float(), dtype=np.float32).reshape(
                want.shape
            )
            np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)

        # Whatever shape the series ended on is the one this deployment keeps.
        settled = series[-1]
        before = np.asarray(
            self._serve(stub, inputs, settled)[0].copyout_float(), dtype=np.float32
        )

        reachable = stub.shape_subgraph_size()
        self.assertGreater(reachable, 0, "there was no shape subgraph to fold")
        for value_info in inputs:
            dynamic = [
                axis
                for axis, dim in enumerate(value_info.type.tensor_type.shape.dim)
                if dim.dim_value <= 0
            ]
            stub.pin_dims(value_info.name, dynamic)
        dropped = stub.fold_shape_subgraph()
        self.assertGreater(dropped, 0, "nothing was folded, so nothing was tested")
        self.assertEqual(stub.shape_subgraph_size(), reachable - dropped)

        after = np.asarray(
            self._serve(stub, inputs, settled)[0].copyout_float(), dtype=np.float32
        )
        np.testing.assert_array_equal(before, after)

    def test_a_pinned_dimension_refuses_another_size(self):
        """Folding rests on the pinned dimension never being given another size.

        Whatever the fold worked out is written into the graph for good, so a
        pinned dimension changing afterwards would leave a stale answer behind
        rather than a wrong one that could be noticed. It has to be refused.
        A dimension left alone is unaffected and still moves.
        """
        model = sequence_model()
        inputs = list(model.graph.input)
        stub = OnnxStub(model, backend.cpu_runtime())
        self._serve(stub, inputs, {"batch": 2, "length": 5})

        stub.pin_dims("x", [0])
        stub.fold_shape_subgraph()

        with self.assertRaises(RuntimeError):
            stub.set_input([[3, 5, 32]])
        # The length was not pinned, so asking for another one is still fine.
        stub.set_input([[2, 9, 32]])
        self.assertEqual(
            next(iter(stub.outputs.values())).shape(), [2, 9, 32]
        )


if __name__ == "__main__":
    unittest.main()
