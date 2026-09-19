"""Which parts of a shape computation the CUDA backend cannot yet run.

The decision this answers: how much kernel work stands between here and a real
model running on the GPU. Asked of graphs built here rather than of the uploaded
models, so that a failure names one missing piece instead of whichever piece the
model happened to reach first -- and so it can be asked before anything is
uploaded at all.

Each stage is attempted on its own and its failure recorded, so one run reports
the whole list. Nothing is fixed here; this only reports.

A stage can also be named on the command line, in which case it is the only one
run. Under compute-sanitizer that is the difference between working and not: a
CudaRuntime preallocates a workspace the moment it is constructed, and eight of
them alongside the sanitizer's own reservation exhausts the card -- which
reports as `cudaErrorMemoryAllocation` at cuda_runtime.cc:86 and looks, at a
glance, exactly like a fault in the kernels being checked.

    python3 probe_cuda_gaps.py --list
    compute-sanitizer python3 probe_cuda_gaps.py int64_arithmetic
"""

import json
import sys
import traceback

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from pyinfinitensor.onnx import OnnxStub, backend


def info(name, dims, dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, dims)


def constant(name, values):
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name)


def model_of(nodes, inputs, outputs, initializers=()):
    return helper.make_model(
        helper.make_graph(
            nodes,
            "cuda_gap_probe",
            [info(*x) for x in inputs],
            [info(*x) for x in outputs],
            list(initializers),
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=9,
    )


# A stage named on the command line is the only one run. Everything else is
# skipped without building a model or a runtime, so nothing else touches the
# card.
WANTED = [a for a in sys.argv[1:] if not a.startswith("-")]

# `--list` names the stages and runs none of them, so it can be asked on a
# machine with no GPU.
LIST_ONLY = "--list" in sys.argv[1:]


def attempt(name, build, feed_shape=None):
    """Run one stage and say how far it got.

    Import, shape inference, memory planning and execution are separate things
    that can each be the one missing, so the stage that failed is reported
    rather than a single pass/fail for the lot.
    """
    if LIST_ONLY or (WANTED and name not in WANTED):
        return {"stage": name, "reached": "skipped", "ok": None, "error": None}
    record = {"stage": name, "reached": "start", "ok": False, "error": None}
    try:
        model = build()
        record["reached"] = "model_built"
        stub = OnnxStub(model, backend.cuda_runtime())
        record["reached"] = "imported"
        stub.handler.shape_infer()
        record["reached"] = "shape_inferred"
        if feed_shape is not None:
            stub.set_input([list(feed_shape)])
            record["reached"] = "shape_set"
        stub.handler.data_malloc()
        record["reached"] = "allocated"
        for tensor_name, tensor in stub.inputs.items():
            dims = tensor.shape()
            count = int(np.prod(dims)) if dims else 1
            tensor.copyin_numpy(
                np.arange(count, dtype=np.float32).reshape(dims)
            )
        record["reached"] = "input_copied"
        stub.run()
        record["reached"] = "ran"
        outs = {}
        for out_name, tensor in stub.outputs.items():
            outs[out_name] = {"shape": list(tensor.shape())}
        record["outputs"] = outs
        record["ok"] = True
    except Exception as exc:  # noqa: BLE001 -- the message is the finding
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback_tail"] = traceback.format_exc().strip().splitlines()[-1]
    return record


results = []

# Does the runtime exist at all -- a build without CUDA reports it here rather
# than by failing every stage below for the same reason.
#
# This check constructs a runtime of its own, and a runtime preallocates a
# workspace. When one stage has been named the whole point is that exactly one
# runtime is built, so the check stands aside; the named stage would report a
# missing CUDA build just as plainly. `--list` must not touch the card at all.
if not (WANTED or LIST_ONLY):
    try:
        rt = backend.cuda_runtime()
        results.append({"stage": "cuda_runtime", "ok": True, "error": None})
    except Exception as exc:  # noqa: BLE001
        results.append(
            {"stage": "cuda_runtime", "ok": False,
             "error": f"{type(exc).__name__}: {exc}"}
        )
        print(json.dumps({"probe": "cuda_gaps", "results": results}, indent=2))
        sys.exit(0)

# 1. Shape on its own. The CPU has ShapeNaive_CPU; the CUDA REGISTER_KERNEL list
# has no Shape, so this is the one gap already known before running anything.
results.append(
    attempt(
        "shape_only",
        lambda: model_of(
            [helper.make_node("Shape", ["x"], ["s"])],
            [("x", ["batch", 4, 8])],
            [("s", [3], TensorProto.INT64)],
        ),
        feed_shape=[2, 4, 8],
    )
)

# 2. Gather over an int64 shape list. Gather has a CUDA kernel, but a shape
# subgraph asks it for int64 rather than the float a data path would.
results.append(
    attempt(
        "shape_gather_int64",
        lambda: model_of(
            [
                helper.make_node("Shape", ["x"], ["s"]),
                helper.make_node("Gather", ["s", "zero"], ["batch"], axis=0),
            ],
            [("x", ["batch", 4, 8])],
            [("batch", [], TensorProto.INT64)],
            [constant("zero", 0)],
        ),
        feed_shape=[2, 4, 8],
    )
)

# 3. Integer arithmetic on a dimension. The unknown that decides the day's
# scope: element_wise.cu may only be instantiated for float, in which case the
# whole integer half of every shape subgraph is missing on the GPU.
results.append(
    attempt(
        "int64_arithmetic",
        lambda: model_of(
            [
                helper.make_node("Shape", ["x"], ["s"]),
                helper.make_node("Gather", ["s", "zero"], ["batch"], axis=0),
                helper.make_node("Mul", ["batch", "two"], ["doubled"]),
                helper.make_node("Add", ["doubled", "one"], ["out"]),
            ],
            [("x", ["batch", 4, 8])],
            [("out", [], TensorProto.INT64)],
            [constant("zero", 0), constant("two", 2), constant("one", 1)],
        ),
        feed_shape=[2, 4, 8],
    )
)

# 4. Unsqueeze and Concat joining dimensions into a target, which is how an
# exporter builds one.
results.append(
    attempt(
        "unsqueeze_concat",
        lambda: model_of(
            [
                helper.make_node("Shape", ["x"], ["s"]),
                helper.make_node("Gather", ["s", "zero"], ["batch"], axis=0),
                helper.make_node("Unsqueeze", ["batch", "zero"], ["batch_1d"]),
                helper.make_node("Concat", ["batch_1d", "tail"], ["target"], axis=0),
            ],
            [("x", ["batch", 4, 8])],
            [("target", [2], TensorProto.INT64)],
            [constant("zero", 0), constant("tail", [32])],
        ),
        feed_shape=[2, 4, 8],
    )
)

# 5. The whole chain the project is about: a Reshape whose target the graph
# works out. This is the通过标准 path, on the GPU.
results.append(
    attempt(
        "full_chain_reshape",
        lambda: model_of(
            [
                helper.make_node("Shape", ["x"], ["s"]),
                helper.make_node("Gather", ["s", "zero"], ["batch"], axis=0),
                helper.make_node("Unsqueeze", ["batch", "zero"], ["batch_1d"]),
                helper.make_node("Concat", ["batch_1d", "tail"], ["target"], axis=0),
                helper.make_node("Reshape", ["x", "target"], ["y"]),
            ],
            [("x", ["batch", 4, 8])],
            [("y", ["batch", 32])],
            [constant("zero", 0), constant("tail", [32])],
        ),
        feed_shape=[2, 4, 8],
    )
)

# 6. Cast, which exporters put between an integer dimension and arithmetic.
results.append(
    attempt(
        "cast_int64_float",
        lambda: model_of(
            [
                helper.make_node("Shape", ["x"], ["s"]),
                helper.make_node("Cast", ["s"], ["sf"], to=TensorProto.FLOAT),
            ],
            [("x", ["batch", 4, 8])],
            [("sf", [3])],
        ),
        feed_shape=[2, 4, 8],
    )
)

# 7. Slice over a shape list, which the simplified attention export uses.
results.append(
    attempt(
        "shape_slice",
        lambda: model_of(
            [
                helper.make_node("Shape", ["x"], ["s"]),
                helper.make_node("Slice", ["s", "start", "end"], ["head"]),
            ],
            [("x", ["batch", 4, 8])],
            [("head", [1], TensorProto.INT64)],
            [constant("start", [0]), constant("end", [1])],
        ),
        feed_shape=[2, 4, 8],
    )
)

# Every stage has now been declared, so `--list` can name them. They are the
# names the stages carry, not a second list kept alongside, so the two cannot
# fall out of step.
if LIST_ONLY:
    for r in results:
        print(r["stage"])
    sys.exit(0)

# A name that matches nothing is a typo, and running the whole probe instead of
# the one stage asked for would look under compute-sanitizer exactly like the
# out-of-memory this option exists to avoid. So it stops and says so.
if WANTED:
    known = [r["stage"] for r in results]
    missing = [w for w in WANTED if w not in known]
    if missing:
        print("no such stage: %s" % ", ".join(missing), file=sys.stderr)
        print("stages: %s" % ", ".join(known), file=sys.stderr)
        sys.exit(2)

# A stage that was skipped is left out rather than reported as neither passing
# nor failing, so naming one stage prints just that stage.
shown = [r for r in results if r.get("reached") != "skipped"]

print(json.dumps({"probe": "cuda_gaps", "diagnostic_only": True,
                  "only": WANTED or None, "results": shown}, indent=2))
