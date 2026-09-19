"""Run a model whose input shape is not known until the moment it is given.

An ONNX dimension is either a number or a name. A named one -- `batch`, say --
carries no number at all, so reading shapes with `dim.dim_value` yields a zero
for it and an empty tensor after that. Such a model needs a shape supplied per
inference instead, which is what `set_input` is for.

Three demo models are built in. The first varies its batch size; the second
varies the height and width of a picture, which the model reads back out of a
pooled tensor to restore the two axes it flattened. The second is the harder
case, and the one where a chain that only ever reads a leading dimension will
not do.

The third varies the length of a sequence, and its shape chain reads a
dimension the model declared as a number alongside two it did not. The part
reading the fixed one is the same under every shape the model may be given, so
it can be worked out once instead of on every inference; passing `--fold` folds
exactly that part away and reports how much of the chain was left.

Usage:
    python dynamic_shape_inference.py
    python dynamic_shape_inference.py batch                # just the first
    python dynamic_shape_inference.py image                # just the second
    python dynamic_shape_inference.py sequence             # just the third
    python dynamic_shape_inference.py sequence --fold      # and fold it first
    python dynamic_shape_inference.py model.onnx           # a model of your own
    python dynamic_shape_inference.py model.onnx batch=4 height=32 width=32

For a model of your own, name each dynamic dimension and the sizes to try for
it. A bare number stands for every named dimension at once, which is what a
model with a single dynamic dimension wants. Dimensions the model gives a
number to are kept as they are.
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from pyinfinitensor import backend
from pyinfinitensor.onnx import OnnxStub


def batch_model():
    """A model that reads its own batch size, works on it, and restores it.

    Neither `Reshape` can be worked out when the graph is built: both targets
    come from the `Shape` of the input, which is a different number on every
    inference.
    """
    weight = (np.arange(36, dtype=np.float32).reshape(6, 6) / 18 - 1) / 6
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Shape", ["x"], ["s"]),
                helper.make_node("Gather", ["s", "first"], ["batch"], axis=0),
                helper.make_node("Unsqueeze", ["batch", "zero"], ["batch_1d"]),
                helper.make_node("Concat", ["batch_1d", "six"], ["flat"], axis=0),
                helper.make_node("Reshape", ["x", "flat"], ["rows"]),
                helper.make_node("MatMul", ["rows", "weight"], ["hidden"]),
                helper.make_node("Relu", ["hidden"], ["activated"]),
                helper.make_node(
                    "Concat", ["batch_1d", "two", "three"], ["restored"], axis=0
                ),
                helper.make_node("Reshape", ["activated", "restored"], ["y"]),
            ],
            "dynamic_batch",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 2, 3])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 2, 3])],
            [
                numpy_helper.from_array(np.asarray(0, np.int64), "first"),
                numpy_helper.from_array(np.asarray([0], np.int64), "zero"),
                numpy_helper.from_array(np.asarray([6], np.int64), "six"),
                numpy_helper.from_array(np.asarray([2], np.int64), "two"),
                numpy_helper.from_array(np.asarray([3], np.int64), "three"),
                numpy_helper.from_array(weight, "weight"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )


def image_model():
    """A model over pictures of no fixed size.

    A convolution and a pooling both change the spatial size, so by the time
    the shape subgraph reads it, the height and the width are numbers no part
    of the graph was given. Both are taken out separately, the two of them are
    flattened into one axis so a softmax can run along a length only this chain
    knows, and both are then put back.
    """
    rng = np.random.default_rng(20260831)
    first = (rng.standard_normal((8, 3, 3, 3)) / 5).astype(np.float32)
    second = (rng.standard_normal((4, 8, 3, 3)) / 5).astype(np.float32)
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Conv", ["x", "w1"], ["c1"], pads=[1, 1, 1, 1]),
                helper.make_node("Relu", ["c1"], ["r1"]),
                helper.make_node(
                    "MaxPool", ["r1"], ["p"], kernel_shape=[2, 2], strides=[2, 2]
                ),
                helper.make_node("Conv", ["p", "w2"], ["c2"], pads=[1, 1, 1, 1]),
                helper.make_node("Relu", ["c2"], ["r2"]),
                helper.make_node("Shape", ["r2"], ["s"]),
                helper.make_node("Gather", ["s", "i0"], ["n"], axis=0),
                helper.make_node("Gather", ["s", "i2"], ["h"], axis=0),
                helper.make_node("Gather", ["s", "i3"], ["w"], axis=0),
                helper.make_node("Unsqueeze", ["n", "zero"], ["n1"]),
                helper.make_node("Unsqueeze", ["h", "zero"], ["h1"]),
                helper.make_node("Unsqueeze", ["w", "zero"], ["w1d"]),
                helper.make_node("Concat", ["n1", "four", "minus1"], ["flat"], axis=0),
                helper.make_node("Reshape", ["r2", "flat"], ["rows"]),
                helper.make_node("Softmax", ["rows"], ["soft"], axis=-1),
                helper.make_node(
                    "Concat", ["n1", "four", "h1", "w1d"], ["back"], axis=0
                ),
                helper.make_node("Reshape", ["soft", "back"], ["y"]),
            ],
            "dynamic_image",
            [
                helper.make_tensor_value_info(
                    "x", TensorProto.FLOAT, ["batch", 3, "height", "width"]
                )
            ],
            [
                helper.make_tensor_value_info(
                    "y", TensorProto.FLOAT, ["batch", 4, "out_h", "out_w"]
                )
            ],
            [
                numpy_helper.from_array(first, "w1"),
                numpy_helper.from_array(second, "w2"),
                numpy_helper.from_array(np.asarray(0, np.int64), "i0"),
                numpy_helper.from_array(np.asarray(2, np.int64), "i2"),
                numpy_helper.from_array(np.asarray(3, np.int64), "i3"),
                numpy_helper.from_array(np.asarray([0], np.int64), "zero"),
                numpy_helper.from_array(np.asarray([4], np.int64), "four"),
                numpy_helper.from_array(np.asarray([-1], np.int64), "minus1"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )


def sequence_model():
    """A model over sequences, whose shape chain reads a fixed dimension too.

    The feature width is a number the model was given -- 32 -- while the batch
    and the length of the sequence are not. The chain splits that width into a
    number of heads and a width per head, works on each head separately, and
    puts the two back together, which is what an attention export does around
    its projections.

    The part of the chain that reads the feature width is the same under every
    shape this model may legally be given, because a dimension declared as a
    number cannot be changed. The part reading the batch and the length is not.
    So this model has both kinds in one chain, which is what makes it the one
    worth folding.
    """
    rng = np.random.default_rng(20260901)
    weight = (rng.standard_normal((8, 8)) / 4).astype(np.float32)
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Shape", ["x"], ["s"]),
                helper.make_node("Gather", ["s", "i0"], ["batch"], axis=0),
                helper.make_node("Gather", ["s", "i1"], ["length"], axis=0),
                # The one dimension of the three that the model gave a number.
                helper.make_node("Gather", ["s", "i2"], ["width"], axis=0),
                # Whatever is left of the width once it is split between heads.
                # Both operands are settled, so this arithmetic is too.
                helper.make_node("Div", ["width", "heads"], ["per_head"]),
                helper.make_node("Unsqueeze", ["batch", "zero"], ["batch_1d"]),
                helper.make_node("Unsqueeze", ["length", "zero"], ["length_1d"]),
                helper.make_node("Unsqueeze", ["per_head", "zero"], ["per_head_1d"]),
                helper.make_node(
                    "Concat",
                    ["batch_1d", "length_1d", "heads_1d", "per_head_1d"],
                    ["split"],
                    axis=0,
                ),
                helper.make_node("Reshape", ["x", "split"], ["by_head"]),
                helper.make_node("MatMul", ["by_head", "weight"], ["mixed"]),
                helper.make_node("Relu", ["mixed"], ["activated"]),
                helper.make_node(
                    "Concat",
                    ["batch_1d", "length_1d", "width_1d"],
                    ["joined"],
                    axis=0,
                ),
                helper.make_node("Reshape", ["activated", "joined"], ["y"]),
            ],
            "dynamic_sequence",
            [
                helper.make_tensor_value_info(
                    "x", TensorProto.FLOAT, ["batch", "length", 32]
                )
            ],
            [
                helper.make_tensor_value_info(
                    "y", TensorProto.FLOAT, ["batch", "length", 32]
                )
            ],
            [
                numpy_helper.from_array(np.asarray(0, np.int64), "i0"),
                numpy_helper.from_array(np.asarray(1, np.int64), "i1"),
                numpy_helper.from_array(np.asarray(2, np.int64), "i2"),
                numpy_helper.from_array(np.asarray(4, np.int64), "heads"),
                numpy_helper.from_array(np.asarray([0], np.int64), "zero"),
                numpy_helper.from_array(np.asarray([4], np.int64), "heads_1d"),
                numpy_helper.from_array(np.asarray([32], np.int64), "width_1d"),
                numpy_helper.from_array(weight, "weight"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )


def shape_of(value_info, sizes, fallback):
    """The shape to ask for, with every dimension lacking a number filled in.

    A dimension holds a number or it does not; asking which field is set is the
    whole test. Reading `dim_value` off one that carries a name instead returns
    a zero, which is a shape a tensor can genuinely have and so passes quietly.

    A named dimension takes its size from `sizes` under that name, and
    `fallback` otherwise. Giving every named dimension the same number is only
    right when there is one of them: a picture whose height and width both came
    from a batch size would be square by accident.
    """
    shape = []
    for d in value_info.type.tensor_type.shape.dim:
        if d.HasField("dim_value"):
            shape.append(d.dim_value)
        else:
            shape.append(sizes.get(d.dim_param, fallback))
    return shape


def reference(model, feeds):
    """What onnxruntime makes of the same inputs, if it is installed."""
    try:
        from onnxruntime import InferenceSession
    except ImportError:
        return None
    session = InferenceSession(model.SerializeToString())
    return session.run(None, feeds)[0]


# What to try for each built-in model. The sizes are picked for what they
# exercise: a repeat takes the path where nothing has to be laid out again, and
# a shape smaller than an earlier one is where a buffer kept from the larger one
# would still read as big enough.
DEMOS = {
    "batch": (
        batch_model,
        [{"batch": n} for n in (1, 4, 4, 16, 2)],
    ),
    "image": (
        image_model,
        [
            {"batch": 1, "height": 8, "width": 8},
            {"batch": 1, "height": 8, "width": 8},
            {"batch": 1, "height": 8, "width": 12},
            {"batch": 2, "height": 12, "width": 8},
            {"batch": 1, "height": 4, "width": 4},
        ],
    ),
    "sequence": (
        sequence_model,
        [
            {"batch": 1, "length": 1},
            {"batch": 2, "length": 7},
            {"batch": 8, "length": 3},
            {"batch": 3, "length": 16},
            {"batch": 1, "length": 1},
        ],
    ),
}


def parse(argv):
    """Work out which models to run and at which sizes.

    Returns a list of (label, model, schedule) where a schedule is a list of
    dimension-name to size mappings, one per inference.
    """
    args = [a for a in argv[1:] if a not in ("--fold", "--pin")]
    if args and args[0] in DEMOS:
        build, schedule = DEMOS[args[0]]
        return [(args[0], build(), schedule)]
    if not args:
        return [(name, build(), schedule) for name, (build, schedule) in DEMOS.items()]

    model = onnx.load(args[0])
    named, bare = {}, []
    for arg in args[1:]:
        if "=" in arg:
            name, _, value = arg.partition("=")
            named.setdefault(name, []).append(int(value))
        else:
            bare.append(int(arg))
    if named:
        # Each named dimension is stepped through its own sizes together with
        # the others, so the shortest list decides how many inferences run.
        rounds = min(len(v) for v in named.values())
        schedule = [{k: v[i] for k, v in named.items()} for i in range(rounds)]
    else:
        schedule = [{"": n} for n in bare or [1, 4, 16]]
    return [(args[0], model, schedule)]


def infer_once(stub, model, inputs, sizes, rng):
    """Run one inference at `sizes`, and say how the answer compares.

    Returns the line to print. Handing the graph a shape re-infers every shape
    downstream of it, including whatever a Reshape reads from a Shape subgraph.
    """
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
    got = np.asarray(output.copyout_float(), dtype=np.float32)
    # An unnamed size stands for every dynamic dimension at once, so there is
    # no name worth printing beside it.
    asked = ", ".join(f"{k}={v}" if k else str(v) for k, v in sizes.items())
    line = f"[{asked}]: {name} has shape {output.shape()}"

    want = reference(model, feeds)
    if want is not None and want.size == got.size:
        error = float(np.abs(got.reshape(want.shape) - want).max())
        line += f", differs from onnxruntime by at most {error:.2e}"
    return line


def run(label, model, schedule, fold=False):
    inputs = list(model.graph.input)
    stub = OnnxStub(model, backend.cpu_runtime())
    if fold:
        # What the chain costs, counting only the operators that describe
        # shapes, since those are the ones folding can reach.
        before = stub.shape_subgraph_size()
        dropped = stub.fold_shape_subgraph()
        print(
            f"{label:>8}: the shape subgraph holds {before} operators, "
            f"{dropped} of which are the same under every shape this model may "
            f"be given and were worked out once; {before - dropped} remain to "
            f"be computed per inference"
        )
    rng = np.random.default_rng(0)
    for sizes in schedule:
        print(f"{label:>6} " + infer_once(stub, model, inputs, sizes, rng))


def run_pinned(label, model, schedule):
    """The same model, deployed at one shape for good.

    A dimension the export left dynamic and the deployment pins is knowledge
    the simplifier could not have had, since it ran before the deployment
    existed. So this is where folding has something left to remove.
    """
    inputs = list(model.graph.input)
    stub = OnnxStub(model, backend.cpu_runtime())
    rng = np.random.default_rng(0)

    # Serve the first shape of the schedule, and only that one.
    sizes = schedule[0]
    print(f"{label:>8}: " + infer_once(stub, model, inputs, sizes, rng))

    loose = stub.shape_subgraph_size()
    for value_info in inputs:
        name = value_info.name
        dynamic = [
            axis
            for axis, dim in enumerate(value_info.type.tensor_type.shape.dim)
            if dim.dim_value <= 0
        ]
        if dynamic:
            stub.pin_dims(name, dynamic)
    pinned = stub.fold_shape_subgraph()
    print(
        f"{label:>8}: pinned at this shape, {pinned} of those {loose} shape "
        f"operators became the same under every remaining inference; "
        f"{stub.shape_subgraph_size()} remain"
    )

    # The answers must not move, and the pinned shape is the only one left to
    # ask for.
    print(f"{label:>8}: " + infer_once(stub, model, inputs, sizes, rng))


def main(argv):
    fold = "--fold" in argv
    pin = "--pin" in argv
    for label, model, schedule in parse(argv):
        if pin:
            run_pinned(label, model, schedule)
        else:
            run(label, model, schedule, fold=fold)


if __name__ == "__main__":
    main(sys.argv)
