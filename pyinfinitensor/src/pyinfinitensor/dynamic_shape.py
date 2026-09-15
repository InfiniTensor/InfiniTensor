"""ONNX dimension contracts and conservative shape-subgraph partial evaluation.

Runtime evaluation lives in C++ (core/shape_executor.cc). This module only folds
values proved constant from the model declaration, never placeholder dimensions.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from onnx import helper, numpy_helper, shape_inference


@dataclass(frozen=True)
class InputDim:
    kind: str
    value: object = None

    @classmethod
    def from_proto(cls, dim):
        if dim.HasField("dim_value"):
            return cls("fixed", dim.dim_value)
        if dim.HasField("dim_param") and dim.dim_param:
            return cls("symbolic", dim.dim_param)
        return cls("unknown")


def input_specs(model):
    weights = {tensor.name for tensor in model.graph.initializer}
    return {
        item.name: tuple(
            InputDim.from_proto(dim) for dim in item.type.tensor_type.shape.dim
        )
        for item in model.graph.input
        if item.name not in weights
    }


def validate_shapes(specs, shapes):
    if len(shapes) != len(specs):
        raise ValueError(
            "inputShapes must contain one shape per model input; expected "
            f"{len(specs)}, got {len(shapes)}"
        )
    symbols, validated = {}, []
    for (name, dims), shape in zip(specs.items(), shapes):
        if len(shape) != len(dims):
            raise ValueError(f"{name}: expected rank {len(dims)}, got {len(shape)}")
        result = []
        for axis, (dim, value) in enumerate(zip(dims, shape)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise ValueError(f"{name}[{axis}]: dimension must be an integer")
            value = int(value)
            if value <= 0 or value > np.iinfo(np.int32).max:
                raise ValueError(f"{name}[{axis}]: dimension must be positive int32")
            if dim.kind == "fixed" and value != dim.value:
                raise ValueError(
                    f"{name}[{axis}]: fixed dimension {dim.value}, got {value}"
                )
            if dim.kind == "symbolic":
                previous = symbols.setdefault(dim.value, value)
                if value != previous:
                    raise ValueError(
                        f"symbol {dim.value}: conflicting dimensions {previous} and {value}"
                    )
            result.append(value)
        validated.append(result)
    return validated


def shape_dependencies(model):
    producers = {name: node for node in model.graph.node for name in node.output}
    required = set()

    def visit(name):
        if not name or name in required:
            return
        required.add(name)
        node = producers.get(name)
        if node is not None and node.op_type != "Shape":
            for source in node.input:
                visit(source)

    for node in model.graph.node:
        if node.op_type == "Shape":
            visit(node.output[0])
        elif node.op_type == "Reshape" and len(node.input) == 2:
            visit(node.input[1])
    constants = {t.name for t in model.graph.initializer}
    constants.update(
        name for n in model.graph.node if n.op_type == "Constant" for name in n.output
    )
    supported = {
        "Gather",
        "Unsqueeze",
        "Squeeze",
        "Concat",
        "Cast",
        "Identity",
        "Reshape",
    }
    while True:
        previous = len(required)
        for node in model.graph.node:
            if (
                node.op_type in supported
                and any(n in required for n in node.input)
                and all(n in required or n in constants for n in node.input)
            ):
                for name in node.output:
                    visit(name)
        if len(required) == previous:
            break
    return required


def fold_shape_subgraphs(model, enabled=True):
    required = shape_dependencies(model)
    candidates = [
        node
        for node in model.graph.node
        if node.op_type != "Constant" and any(n in required for n in node.output)
    ]
    stats = {
        "nodes_before": len(candidates),
        "nodes_after": len(candidates),
        "folded_nodes": 0,
        "eliminated_nodes": 0,
    }
    if not enabled or not candidates:
        return model, stats
    # Inferred metadata can prove a channel constant even if batch/H/W vary.
    try:
        inferred = shape_inference.infer_shapes(model)
    except Exception:
        inferred = model
    shapes = {
        value.name: [
            dim.dim_value if dim.HasField("dim_value") else None
            for dim in value.type.tensor_type.shape.dim
        ]
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
        if value.type.tensor_type.HasField("shape")
    }
    values = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    shapes.update({t.name: list(t.dims) for t in model.graph.initializer})
    replacements = {}
    for node in model.graph.node:
        if not any(name in required for name in node.output):
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        result = None
        try:
            args = [values.get(name) for name in node.input]
            if node.op_type == "Constant" and "value" in attrs:
                result = numpy_helper.to_array(attrs["value"])
            elif node.op_type == "Shape" and node.input[0] in shapes:
                rank = len(shapes[node.input[0]])
                result = np.array(shapes[node.input[0]], dtype=object)[
                    slice(attrs.get("start", 0), attrs.get("end", rank))
                ]
            elif args and all(arg is not None for arg in args):
                if node.op_type == "Gather":
                    result = np.take(
                        args[0], args[1].astype(np.int64), axis=attrs.get("axis", 0)
                    )
                elif node.op_type == "Unsqueeze":
                    axes = args[1].tolist() if len(args) > 1 else attrs["axes"]
                    result = np.expand_dims(args[0], tuple(axes))
                elif node.op_type == "Squeeze":
                    axes = args[1].tolist() if len(args) > 1 else attrs.get("axes")
                    result = np.squeeze(args[0], None if axes is None else tuple(axes))
                elif node.op_type == "Concat":
                    result = np.concatenate(args, axis=attrs["axis"])
                elif node.op_type == "Cast":
                    if all(v is not None for v in args[0].flat):
                        result = args[0].astype(
                            helper.tensor_dtype_to_np_dtype(attrs["to"])
                        )
                elif node.op_type == "Identity":
                    result = args[0]
        except (ValueError, TypeError, IndexError, KeyError):
            # Leave invalid/unsupported nodes for normal frontend diagnostics.
            continue
        if result is None:
            continue
        result = np.asarray(result)
        if result.size > 4096:
            continue
        values[node.output[0]] = result
        if all(value is not None for value in result.flat):
            if result.dtype == object:
                result = result.astype(np.int64)
            replacements[node.output[0]] = numpy_helper.from_array(
                result, node.output[0]
            )
            if node.op_type != "Constant":
                stats["folded_nodes"] += 1
    kept = [node for node in model.graph.node if node.output[0] not in replacements]
    # Remove obsolete shape branches, but never remove a graph output or data op.
    while True:
        used = {name for node in kept for name in node.input}
        used.update(item.name for item in model.graph.output)
        reduced = [
            node
            for node in kept
            if not all(n in required and n not in used for n in node.output)
        ]
        if len(reduced) == len(kept):
            break
        stats["eliminated_nodes"] += len(kept) - len(reduced)
        kept = reduced
    del model.graph.node[:]
    model.graph.node.extend(kept)
    model.graph.initializer.extend(replacements.values())
    used = {name for node in kept for name in node.input}
    used.update(v.name for v in list(model.graph.input) + list(model.graph.output))
    initializers = [t for t in model.graph.initializer if t.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    # Folded intermediates are now initializers, not duplicated value_info.
    infos = [v for v in model.graph.value_info if v.name not in replacements]
    del model.graph.value_info[:]
    model.graph.value_info.extend(infos)
    stats["nodes_after"] = sum(
        node.op_type != "Constant" and any(name in required for name in node.output)
        for node in kept
    )
    return model, stats
