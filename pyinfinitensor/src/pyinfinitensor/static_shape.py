"""Bounded import-time evaluation of constant shape and index expressions."""

import math

import numpy as np
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


_LIMIT = 65536
_BROADCAST = {"Add", "Sub", "Mul", "Div", "Mod", "Equal", "Less", "Greater", "Where"}
_OPS = _BROADCAST | {
    "Cast", "Concat", "ConstantOfShape", "Exp", "Expand", "Gather", "Identity",
    "Range", "Reshape", "Slice", "Squeeze", "Unsqueeze",
}


def evaluate_static_shape_node(node, constants, tensors, opsets):
    """Return an array only when the node needs metadata or small constants."""
    if node.domain or len(node.output) != 1:
        return None
    op = node.op_type
    if op == "Shape":
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        dims = np.asarray(tensors[node.input[0]].shape(), dtype=np.int64)
        return dims[attrs.get("start", 0):attrs.get("end")]
    if op not in _OPS:
        return None
    names = [name for name in node.input if name]
    if not names or any(name not in constants for name in names):
        return None
    values = [constants[name] for name in names]
    if any(t.data_location == TensorProto.EXTERNAL or math.prod(t.dims) > _LIMIT
           for t in values):
        return None
    arrays = [numpy_helper.to_array(t) for t in values]
    if any(a.dtype.kind not in "biuf" for a in arrays):
        return None
    if op == "Exp" and arrays[0].size != 1:
        return None

    if op in _BROADCAST:
        count = math.prod(np.broadcast_shapes(*(a.shape for a in arrays)))
    elif op in {"ConstantOfShape", "Expand"}:
        shape = tuple(int(n) for n in arrays[-1].ravel())
        if any(n < 0 or n > _LIMIT for n in shape):
            return None
        if op == "Expand":
            shape = np.broadcast_shapes(arrays[0].shape, shape)
        count = math.prod(shape)
    elif op == "Gather":
        data, indices = arrays
        axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 0) % data.ndim
        count = math.prod(data.shape[:axis] + data.shape[axis + 1:]) * indices.size
    elif op == "Range":
        start, end, step = (a.item() for a in arrays)
        if not all(math.isfinite(v) for v in (start, end, step)) or step == 0:
            return None
        distance = (end - start) / step
        if not math.isfinite(distance) or distance > _LIMIT:
            return None
        count = max(0, math.ceil(distance))
    else:
        count = sum(a.size for a in arrays)
    if count > _LIMIT:
        return None
    result = ReferenceEvaluator(node, opsets=opsets).run(None, dict(zip(names, arrays)))[0]
    if result.size > _LIMIT:
        raise ValueError(f"Static shape evaluation exceeded its element limit: {node.name}")
    return np.asarray(result)
