"""Host shape program + persistent InfiniTensor data graph (ONNX opsets 13-18).

Only shape/control tensors are evaluated with NumPy. Float data operations run
in the selected InfiniTensor runtime; ONNX Runtime is never used here.
"""

import copy
import math
import numpy as np
from onnx import helper, numpy_helper, TensorProto
from . import backend


_CONTROL = {"Shape", "Gather", "Unsqueeze", "Squeeze", "Concat", "Cast", "Constant"}
_DATA = {"Reshape", "Identity", "Relu", "Add", "Sub", "Mul", "MatMul", "Transpose"}


def _attrs(node):
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def _evaluate(node, values, shapes):
    a = _attrs(node)
    x = [values.get(n) for n in node.input]
    op = node.op_type
    if op == "Constant":
        if "value" not in a:
            raise ValueError("Constant requires a tensor value attribute")
        return numpy_helper.to_array(a["value"]).copy()
    if op == "Shape":
        shape = shapes[node.input[0]]
        return np.asarray(
            shape[a.get("start", 0) : a.get("end", len(shape))], dtype=np.int64
        )
    if any(v is None for v in x):
        raise ValueError(f"{op}: shape computation depends on unsupported data values")
    if op == "Gather":
        if x[1].dtype not in (np.dtype("int32"), np.dtype("int64")):
            raise ValueError("Gather indices must be int32/int64")
        return np.asarray(np.take(x[0], x[1], axis=a.get("axis", 0)))
    if op in ("Unsqueeze", "Squeeze"):
        axes = x[1] if len(x) > 1 else a.get("axes")
        if axes is not None:
            axes_array = np.asarray(axes)
            if axes_array.dtype != np.int64 or axes_array.ndim != 1:
                raise ValueError(f"{op}: axes must be a rank-one int64 tensor")
            axes = tuple(int(i) for i in axes_array)
            rank = x[0].ndim + (len(axes) if op == "Unsqueeze" else 0)
            normalized = tuple(i + rank if i < 0 else i for i in axes)
            if len(set(normalized)) != len(axes) or any(
                i < 0 or i >= rank for i in normalized
            ):
                raise ValueError(f"{op}: duplicate or out-of-range axes")
            axes = normalized
        if op == "Unsqueeze":
            if axes is None:
                raise ValueError("Unsqueeze requires axes")
            return np.expand_dims(x[0], axes)
        return np.asarray(np.squeeze(x[0], axis=axes))
    if op == "Concat":
        if len({v.dtype for v in x}) != 1:
            raise ValueError("Concat input dtypes must match")
        return np.concatenate(x, axis=a["axis"])
    if op == "Cast":
        dtype = helper.tensor_dtype_to_np_dtype(a["to"])
        if np.dtype(dtype).kind not in "iufb":
            raise ValueError("Unsupported shape Cast dtype")
        return x[0].astype(dtype)
    raise ValueError(f"Unsupported shape operator: {op}")


def _reshape(shape, target, allowzero=0):
    if target.dtype != np.int64 or target.ndim != 1:
        raise ValueError("Reshape target must be a rank-one int64 tensor")
    dims = [int(d) for d in target]
    if allowzero:
        raise ValueError("Reshape allowzero=1 is not supported by this backend")
    if dims.count(-1) > 1 or any(d < -1 for d in dims):
        raise ValueError("Reshape allows at most one -1 and no values below -1")
    for i, d in enumerate(dims):
        if d == 0:
            if i >= len(shape):
                raise ValueError("Reshape zero index exceeds input rank")
            dims[i] = shape[i]
    size = math.prod(shape)
    known = math.prod(d for d in dims if d != -1)
    if -1 in dims:
        if known == 0 or size % known:
            raise ValueError("Reshape inferred dimension is not integral")
        dims[dims.index(-1)] = size // known
    if math.prod(dims) != size:
        raise ValueError("Reshape changes the number of elements")
    if any(d <= 0 or d > 2147483647 for d in dims):
        raise ValueError("Zero-sized or oversized dimensions are not supported")
    return dims


class DynamicOnnxStub:
    """Load once, then run(feeds) repeatedly with concrete shapes.

    Fixed, symbolic and anonymous dimensions are retained in input_specs.
    Shape operators execute as a host control program before C++ allocation.
    The data graph and its operator/tensor identities persist across runs.
    """

    def __init__(
        self, model, runtime, *, fold_constants=True, use_naive_allocator=False
    ):
        self.model = copy.deepcopy(model)
        versions = {o.domain: o.version for o in model.opset_import}
        if not 13 <= versions.get("", 0) <= 18:
            raise ValueError("DynamicOnnxStub supports standard ONNX opsets 13-18")
        self.handler = backend.GraphHandler(runtime)
        self.use_naive_allocator = use_naive_allocator
        self.constants = {
            t.name: numpy_helper.to_array(t).copy() for t in model.graph.initializer
        }
        self.input_specs = {}
        self.input_dtypes = {}
        for v in model.graph.input:
            if v.name in self.constants:
                continue
            tt = v.type.tensor_type
            if not tt.HasField("shape"):
                raise ValueError("Unknown input rank is unsupported")
            self.input_specs[v.name] = tuple(
                (
                    d.dim_value
                    if d.HasField("dim_value")
                    else d.dim_param if d.HasField("dim_param") else None
                )
                for d in tt.shape.dim
            )
            self.input_dtypes[v.name] = np.dtype(
                helper.tensor_dtype_to_np_dtype(tt.elem_type)
            )
        self.output_names = [v.name for v in model.graph.output]
        known = set(self.input_specs) | set(self.constants)
        pending = list(model.graph.node)
        self.nodes = []
        while pending:
            ready = [n for n in pending if all(not i or i in known for i in n.input)]
            if not ready:
                raise ValueError("ONNX graph contains missing inputs or a cycle")
            for n in ready:
                if n.domain or n.op_type not in _CONTROL | _DATA or len(n.output) != 1:
                    raise ValueError(
                        f"Unsupported dynamic node: {n.domain}:{n.op_type}"
                    )
                if any(o in known for o in n.output):
                    raise ValueError("Duplicate ONNX value name")
                self.nodes.append(n)
                known.update(n.output)
                pending.remove(n)
        if not set(self.output_names) <= known:
            raise ValueError("Missing graph output")
        self.stats = {
            "shape_nodes_before": sum(n.op_type in _CONTROL for n in self.nodes),
            "folded_nodes": 0,
            "runs": 0,
            "graph_builds": 0,
        }
        self.folded = set()
        if fold_constants:
            for i, n in enumerate(self.nodes):
                # A Shape of constant storage is independent of input bindings.
                if n.op_type in _CONTROL and all(k in self.constants for k in n.input):
                    shapes = {k: v.shape for k, v in self.constants.items()}
                    self.constants[n.output[0]] = _evaluate(n, self.constants, shapes)
                    self.folded.add(i)
        self.stats["folded_nodes"] = len(self.folded)
        self.stats["shape_nodes_after"] = self.stats["shape_nodes_before"] - len(
            self.folded
        )
        self.tensors, self.inputs, self.outputs, self.ops = {}, {}, {}, {}
        self._initialized_weights = set()

    def _validate(self, feeds):
        if set(feeds) != set(self.input_specs):
            raise ValueError(
                f"Expected inputs {sorted(self.input_specs)}, got {sorted(feeds)}"
            )
        symbols = {}
        result = {}
        for name, spec in self.input_specs.items():
            value = np.asarray(feeds[name])
            if value.dtype != self.input_dtypes[name]:
                raise ValueError(
                    f"{name}: expected dtype {self.input_dtypes[name]}, got {value.dtype}"
                )
            if value.ndim != len(spec):
                raise ValueError(f"{name}: expected rank {len(spec)}, got {value.ndim}")
            if value.size > 2147483647 or any(d <= 0 for d in value.shape):
                raise ValueError(f"{name}: empty/oversized tensors are unsupported")
            for declared, actual in zip(spec, value.shape):
                if isinstance(declared, int) and declared != actual:
                    raise ValueError(
                        f"{name}: fixed dimension {declared} cannot change to {actual}"
                    )
                if isinstance(declared, str):
                    if declared in symbols and symbols[declared] != actual:
                        raise ValueError(f"Inconsistent symbolic dimension {declared}")
                    symbols[declared] = actual
            result[name] = np.require(value, requirements=["C"])
        return result

    def _tensor(self, name, value, weight=False):
        if name not in self.tensors:
            t = self.handler.tensor(
                list(value.shape), helper.np_dtype_to_tensor_dtype(value.dtype)
            )
            t.set_weight() if weight else t.set_input()
            self.tensors[name] = t
        else:
            # Scalar control tensors never change rank or storage shape.
            if list(value.shape) != self.tensors[name].shape():
                self.handler.change_shape(list(value.shape), self.tensors[name].fuid())
        return self.tensors[name]

    def run(self, feeds, *, cuda_graph=False):
        feeds = self._validate(feeds)
        values = dict(self.constants)
        # Integer graph inputs can be runtime shape tensors.
        values.update({k: v for k, v in feeds.items() if v.dtype.kind in "iu"})
        shapes = {k: list(v.shape) for k, v in {**self.constants, **feeds}.items()}
        # Validate and solve the entire shape plan before changing live tensors.
        plan = []
        for i, n in enumerate(self.nodes):
            out, op, a = n.output[0], n.op_type, _attrs(n)
            if i in self.folded:
                continue
            if op in _CONTROL:
                values[out] = _evaluate(n, values, shapes)
                shapes[out] = list(values[out].shape)
                continue
            s = shapes[n.input[0]]
            if op == "Reshape":
                if n.input[1] not in values:
                    raise ValueError(
                        "Reshape target must come from a host shape tensor"
                    )
                shape = _reshape(s, values[n.input[1]], a.get("allowzero", 0))
            elif op in ("Add", "Sub", "Mul"):
                shape = list(np.broadcast_shapes(s, shapes[n.input[1]]))
            elif op == "MatMul":
                b = shapes[n.input[1]]
                if len(s) != 2 or len(b) != 2 or s[1] != b[0]:
                    raise ValueError(
                        "Dynamic MatMul currently requires compatible rank-two tensors"
                    )
                shape = [s[0], b[1]]
            elif op == "Transpose":
                perm = list(a.get("perm", reversed(range(len(s)))))
                if sorted(perm) != list(range(len(s))):
                    raise ValueError("Invalid transpose permutation")
                shape = [s[j] for j in perm]
            else:
                shape = list(s)
            if math.prod(shape) > 2147483647:
                raise ValueError("Output exceeds backend integer size limit")
            shapes[out] = shape
            plan.append((i, n, shape, a))
        for name, value in feeds.items():
            self.inputs[name] = self._tensor(name, value)
        for i, n, shape, a in plan:
            ins = []
            for name in n.input[:1] if n.op_type == "Reshape" else n.input:
                if name in values:
                    self._tensor(name, values[name], weight=name in self.constants)
                ins.append(self.tensors[name])
            if any(backend.tensor_dtype(t) != TensorProto.FLOAT for t in ins):
                raise ValueError(
                    "Data graph currently supports float32; integer values belong to the shape program"
                )
            if i not in self.ops:
                op = n.op_type
                if op == "Reshape":
                    out = self.handler.reshape(ins[0], None, shape)
                elif op in ("Add", "Sub", "Mul"):
                    out = getattr(self.handler, op.lower())(ins[0], ins[1], None)
                elif op == "MatMul":
                    out = self.handler.matmul(
                        ins[0],
                        ins[1],
                        None,
                        False,
                        False,
                        None,
                        backend.ActType.Linear,
                        "default",
                    )
                elif op == "Transpose":
                    out = self.handler.transpose(
                        ins[0],
                        None,
                        list(a.get("perm", reversed(range(len(shapes[n.input[0]]))))),
                    )
                else:
                    out = getattr(self.handler, op.lower())(ins[0], None)
                self.tensors[n.output[0]] = out
                self.ops[i] = out.src()
            else:
                if n.op_type == "Reshape":
                    self.ops[i].set_reshape_dims(shape)
                self.ops[i].infer_shape()
            if self.tensors[n.output[0]].shape() != shape:
                raise RuntimeError("Host shape plan disagrees with C++ shape inference")
        for name in self.output_names:
            if name not in self.tensors:
                self._tensor(name, values[name], weight=name in self.constants)
            elif (
                name in values
                and list(values[name].shape) != self.tensors[name].shape()
            ):
                self.handler.change_shape(
                    list(values[name].shape), self.tensors[name].fuid()
                )
            self.tensors[name].set_output()
            self.outputs[name] = self.tensors[name]
        self.handler.data_malloc(self.use_naive_allocator)
        # Copy after memory planning, including runtime control tensors consumed by data ops.
        for name, value in {**values, **feeds}.items():
            if name in self.tensors and self.tensors[name].src() is None:
                if name in self.constants and name in self._initialized_weights:
                    continue
                self.tensors[name].copyin_numpy(np.require(value, requirements=["C"]))
                if name in self.constants:
                    self._initialized_weights.add(name)
        if cuda_graph:
            if not hasattr(self.handler, "run_with_cudagraph"):
                raise ValueError("This build has no CUDA Graph support")
            self.handler.run_with_cudagraph()
        else:
            self.handler.run()
        self.stats["runs"] += 1
        self.stats["graph_builds"] = 1
        return {name: tensor.copyout_numpy() for name, tensor in self.outputs.items()}
