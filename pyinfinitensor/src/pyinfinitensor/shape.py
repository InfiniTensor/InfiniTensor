"""Evaluate ONNX shape-control metadata without reading model activations."""

from functools import reduce
from typing import Dict, List, Sequence

import numpy as np
from onnx import ModelProto, NodeProto, AttributeProto, TensorProto
from onnx.numpy_helper import to_array


def _has_input(node, index):
    return index < len(node.input) and bool(node.input[index])


_SHAPE_EVALUATOR_OPS = {
    "Shape",
    "Gather",
    "Slice",
    "Concat",
    "Unsqueeze",
    "Squeeze",
    "Cast",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Max",
    "Min",
    "Equal",
    "Less",
    "Greater",
    "Where",
    "Range",
    "ConstantOfShape",
    "Expand",
    "Reshape",
    "Identity",
    "Constant",
}


class ShapeValueEvaluator:
    """Evaluate the integer, metadata-only part of an ONNX graph on the CPU.

    This evaluator deliberately never accepts activation values.  Its only
    runtime source is input dimensions, propagated through intermediate tensor
    metadata and exposed through ONNX ``Shape``. It updates Reshape/Expand/Slice
    parameters without treating symbolic dimensions as a permanent ``1``.
    """

    def __init__(self, model: ModelProto):
        self.model = model
        # Retain protobuf wrappers so attribute cache keys remain stable.
        # The importer owns a private, immutable model template.
        self._nodes = tuple(model.graph.node)
        self._control_nodes = _shape_control_node_indexes(model)
        # Only control values are materialized. Weight dimensions are metadata;
        # embedding/linear weight contents must never be loaded into this evaluator.
        required_values = set()
        for index, node in enumerate(self._nodes):
            parameter_indexes = {
                "Reshape": [1],
                "Expand": [1],
                "Slice": [1, 2, 3, 4],
                "Unsqueeze": [1],
                "Squeeze": [1],
            }.get(node.op_type, [])
            required_values.update(
                node.input[i] for i in parameter_indexes if _has_input(node, i)
            )
            if index in self._control_nodes and node.op_type != "Shape":
                required_values.update(node.input)
        self._constant_shapes = {}
        self._attributes = {}
        for index, node in enumerate(self._nodes):
            attributes = {}
            for attr in node.attribute:
                if attr.type == AttributeProto.INT:
                    attributes[attr.name] = attr.i
                elif attr.type == AttributeProto.INTS:
                    attributes[attr.name] = tuple(attr.ints)
                elif attr.type == AttributeProto.TENSOR:
                    if node.op_type == "Constant" and attr.name == "value":
                        self._constant_shapes[node.output[0]] = tuple(attr.t.dims)
                    if index in self._control_nodes or any(
                        name in required_values for name in node.output
                    ):
                        attributes[attr.name] = np.asarray(to_array(attr.t))
            self._attributes[id(node)] = attributes
        self.initial_values = {
            item.name: np.asarray(to_array(item))
            for item in model.graph.initializer
            if item.name in required_values
        }
        for node in self._nodes:
            if node.op_type == "Constant" and node.output[0] in required_values:
                self.initial_values[node.output[0]] = self._attribute(node, "value")
        self._initial_shapes = {
            item.name: tuple(item.dims) for item in model.graph.initializer
        }
        self._initial_shapes.update(self._constant_shapes)
        self._values: Dict[str, np.ndarray] = {}

    def evaluate(self, input_shapes: Dict[str, Sequence[int]]) -> None:
        values = dict(self.initial_values)
        input_shapes = {
            name: tuple(int(dim) for dim in dims) for name, dims in input_shapes.items()
        }
        shapes = dict(self._initial_shapes)
        shapes.update(input_shapes)
        control = self._control_nodes
        pending = list(enumerate(self._nodes))
        progress = True
        while pending and progress:
            progress = False
            next_pending = []
            for index, node in pending:
                try:
                    inferred = self._infer_metadata(node, shapes, values)
                except KeyError:
                    inferred = None
                except (TypeError, ValueError, IndexError) as error:
                    raise ValueError(
                        "shape preparation failed at {} ({}): {}".format(
                            node.name or node.output[0], node.op_type, error
                        )
                    ) from error
                if inferred is not None:
                    for name, shape in zip(node.output, inferred):
                        if name not in shapes:
                            shapes[name] = tuple(shape)
                            progress = True
                if node.op_type == "Constant":
                    continue
                if index not in control:
                    if inferred is None:
                        next_pending.append((index, node))
                    continue
                if node.op_type == "Shape":
                    if not node.input or node.input[0] not in shapes:
                        next_pending.append((index, node))
                        continue
                elif any(name and name not in values for name in node.input):
                    next_pending.append((index, node))
                    continue
                try:
                    result = self._execute(node, values, shapes)
                except (TypeError, ValueError, IndexError):
                    next_pending.append((index, node))
                    continue
                if len(result) != len(node.output):
                    raise ValueError(
                        "shape evaluator output arity mismatch at {}".format(
                            node.name or node.op_type
                        )
                    )
                values.update(zip(node.output, result))
                shapes.update(
                    (name, np.asarray(value).shape)
                    for name, value in zip(node.output, result)
                )
                progress = True
            pending = next_pending
        self._values = values
        self.shapes = shapes

    def _infer_metadata(self, node, shapes, values):
        """Infer tensor dimensions without reading any activation values."""
        op = node.op_type

        def attr(name, default=None):
            return self._attribute(node, name, default)

        if op == "Constant":
            return [self._constant_shapes[node.output[0]]]
        if not node.input or not node.input[0]:
            return None
        x = shapes[node.input[0]]
        if op == "Shape":
            return [(len(x[attr("start", 0) : attr("end", len(x))]),)]
        if op in {
            "Identity",
            "Cast",
            "LayerNormalization",
            "Softmax",
            "Erf",
            "Sqrt",
            "Gelu",
            "Relu",
            "Tanh",
            "Sigmoid",
        }:
            return [x]
        if op in {
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Pow",
            "Equal",
            "Less",
            "Greater",
            "Where",
        }:
            return [np.broadcast_shapes(*(shapes[name] for name in node.input if name))]
        if op == "Gather":
            axis = int(attr("axis", 0))
            if not -len(x) <= axis < len(x):
                raise ValueError("Gather axis out of range")
            axis %= len(x)
            return [x[:axis] + shapes[node.input[1]] + x[axis + 1 :]]
        if op == "MatMul":
            y = shapes[node.input[1]]
            left = (1,) + x if len(x) == 1 else x
            right = y + (1,) if len(y) == 1 else y
            if left[-1] != right[-2]:
                raise ValueError("MatMul contracting dimensions differ")
            out = np.broadcast_shapes(left[:-2], right[:-2]) + (left[-2], right[-1])
            if len(x) == 1:
                out = out[:-2] + out[-1:]
            if len(y) == 1:
                out = out[:-1]
            return [out]
        if op == "Transpose":
            return [tuple(x[i] for i in attr("perm", list(reversed(range(len(x))))))]
        if op in {"Unsqueeze", "Squeeze"}:
            axes = attr("axes")
            if axes is None and len(node.input) > 1:
                axes = values[node.input[1]].reshape(-1).tolist()
            if op == "Squeeze":
                axes = (
                    [i for i, d in enumerate(x) if d == 1]
                    if axes is None
                    else [int(i) % len(x) for i in axes]
                )
                if any(x[i] != 1 for i in axes):
                    raise ValueError("Squeeze requires singleton dimensions")
                return [tuple(d for i, d in enumerate(x) if i not in axes)]
            rank = len(x) + len(axes)
            if any(not -rank <= int(i) < rank for i in axes):
                raise ValueError("Unsqueeze axis out of range")
            axes = sorted(int(i) % rank for i in axes)
            if len(axes) != len(set(axes)):
                raise ValueError("Repeated Unsqueeze axis")
            out = list(x)
            for i in axes:
                out.insert(i, 1)
            return [tuple(out)]
        if op == "Concat":
            axis = int(attr("axis", 0))
            if not -len(x) <= axis < len(x):
                raise ValueError("Concat axis out of range")
            axis %= len(x)
            if any(
                len(shapes[name]) != len(x)
                or any(d != shapes[name][i] for i, d in enumerate(x) if i != axis)
                for name in node.input
            ):
                raise ValueError("Concat non-axis dimensions differ")
            out = list(x)
            out[axis] = sum(shapes[name][axis] for name in node.input)
            return [tuple(out)]
        if op == "Reshape":
            target = [int(d) for d in values[node.input[1]].reshape(-1)]
            if not attr("allowzero", 0):
                target = [x[i] if d == 0 else d for i, d in enumerate(target)]
            if target.count(-1) > 1 or any(d < -1 for d in target):
                raise ValueError("invalid Reshape dimensions")
            count = int(np.prod(x))
            known = int(np.prod([d for d in target if d != -1]))
            if -1 in target:
                if known == 0 or count % known:
                    raise ValueError("invalid Reshape inferred dimension")
                target[target.index(-1)] = count // known
            if int(np.prod(target)) != count:
                raise ValueError("Reshape changes element count")
            return [tuple(target)]
        if op == "Expand":
            return [
                np.broadcast_shapes(
                    x, tuple(int(d) for d in values[node.input[1]].reshape(-1))
                )
            ]
        if op == "Slice":
            starts, ends = (values[name].reshape(-1) for name in node.input[1:3])
            axes = (
                values[node.input[3]].reshape(-1)
                if _has_input(node, 3)
                else range(len(starts))
            )
            steps = (
                values[node.input[4]].reshape(-1)
                if _has_input(node, 4)
                else [1] * len(starts)
            )
            if not len(starts) == len(ends) == len(axes) == len(steps):
                raise ValueError("Slice parameter lengths differ")
            if any(not -len(x) <= int(axis) < len(x) for axis in axes) or len(
                {int(axis) % len(x) for axis in axes}
            ) != len(axes):
                raise ValueError("Invalid or repeated Slice axis")
            out = list(x)
            for start, end, axis, step in zip(starts, ends, axes, steps):
                axis = int(axis) % len(x)
                out[axis] = len(
                    range(*slice(int(start), int(end), int(step)).indices(x[axis]))
                )
            return [tuple(out)]
        return None

    def value(self, name: str) -> List[int]:
        if name not in self._values:
            raise ValueError(
                'dynamic shape value "{}" cannot be evaluated from input metadata; '
                "the path is unsupported or depends on activation data".format(name)
            )
        value = np.asarray(self._values[name])
        if not np.issubdtype(value.dtype, np.integer):
            raise ValueError(
                'shape value "{}" must be integer, got {}'.format(name, value.dtype)
            )
        return [int(item) for item in value.reshape(-1).tolist()]

    def _attribute(self, node: NodeProto, name: str, default=None):
        attributes = self._attributes.get(id(node))
        if attributes is not None:
            return attributes.get(name, default)
        for attr in node.attribute:
            if attr.name == name:
                if attr.type == AttributeProto.INT:
                    return attr.i
                if attr.type == AttributeProto.INTS:
                    return list(attr.ints)
                if attr.type == AttributeProto.TENSOR:
                    return np.asarray(to_array(attr.t))
        return default

    def _execute(self, node, values, input_shapes):
        op = node.op_type
        if op == "Shape":
            shape = list(input_shapes[node.input[0]])
            start = self._attribute(node, "start", 0)
            end = self._attribute(node, "end", len(shape))
            return [np.asarray(shape[start:end], dtype=np.int64)]
        inputs = [values[name] for name in node.input if name]
        if op == "Constant":
            return [self._attribute(node, "value")]
        if op == "Identity":
            return [inputs[0]]
        if op == "Cast":
            onnx_type = int(self._attribute(node, "to"))
            return [inputs[0].astype(_onnx_numpy_dtype(onnx_type), copy=False)]
        if op == "Gather":
            return [
                np.take(
                    inputs[0], inputs[1], axis=int(self._attribute(node, "axis", 0))
                )
            ]
        if op == "Concat":
            return [np.concatenate(inputs, axis=int(self._attribute(node, "axis", 0)))]
        if op == "Unsqueeze":
            axes = self._attribute(node, "axes")
            if axes is None:
                axes = inputs[1].reshape(-1).tolist()
            return [np.expand_dims(inputs[0], tuple(int(x) for x in axes))]
        if op == "Squeeze":
            axes = self._attribute(node, "axes")
            if axes is None and len(inputs) > 1:
                axes = inputs[1].reshape(-1).tolist()
            return [np.squeeze(inputs[0], axis=None if axes is None else tuple(axes))]
        if op == "Slice":
            starts, ends = inputs[1].reshape(-1), inputs[2].reshape(-1)
            axes = (
                values[node.input[3]].reshape(-1)
                if _has_input(node, 3)
                else np.arange(len(starts))
            )
            steps = (
                values[node.input[4]].reshape(-1)
                if _has_input(node, 4)
                else np.ones(len(starts), dtype=np.int64)
            )
            slices = [slice(None)] * inputs[0].ndim
            for start, end, axis, step in zip(starts, ends, axes, steps):
                slices[int(axis)] = slice(int(start), int(end), int(step))
            return [inputs[0][tuple(slices)]]
        if op in {"Add", "Sub", "Mul", "Div", "Max", "Min", "Equal", "Less", "Greater"}:
            if op == "Div":
                left, right = inputs[:2]
                if np.any(right == 0):
                    raise ValueError("division by zero in shape expression")
                if np.issubdtype(left.dtype, np.integer):
                    # ONNX integer division truncates toward zero. Avoid a
                    # float conversion that loses precision for int64 shapes.
                    quotient = np.floor_divide(left, right)
                    correction = ((left < 0) != (right < 0)) & (
                        np.remainder(left, right) != 0
                    )
                    return [quotient + correction.astype(quotient.dtype)]
                return [np.divide(left, right)]
            functions = {
                "Add": np.add,
                "Sub": np.subtract,
                "Mul": np.multiply,
                "Div": np.floor_divide,
                "Max": np.maximum,
                "Min": np.minimum,
                "Equal": np.equal,
                "Less": np.less,
                "Greater": np.greater,
            }
            if op in {"Max", "Min"}:
                return [reduce(functions[op], inputs)]
            return [functions[op](inputs[0], inputs[1])]
        if op == "Where":
            return [np.where(inputs[0], inputs[1], inputs[2])]
        if op == "Range":
            return [
                np.arange(
                    inputs[0].item(),
                    inputs[1].item(),
                    inputs[2].item(),
                    dtype=inputs[0].dtype,
                )
            ]
        if op == "ConstantOfShape":
            fill = self._attribute(node, "value", np.asarray(0, dtype=np.float32))
            return [
                np.full(
                    tuple(int(x) for x in inputs[0].reshape(-1)),
                    fill.reshape(-1)[0],
                    dtype=fill.dtype,
                )
            ]
        if op == "Expand":
            return [
                np.broadcast_to(inputs[0], tuple(int(x) for x in inputs[1].reshape(-1)))
            ]
        if op == "Reshape":
            shapes = {node.input[0]: inputs[0].shape}
            target = self._infer_metadata(node, shapes, values)[0]
            return [np.reshape(inputs[0], target)]
        raise ValueError("unsupported shape op " + op)


def _onnx_numpy_dtype(dtype: int):
    mapping = {
        TensorProto.FLOAT: np.float32,
        TensorProto.DOUBLE: np.float64,
        TensorProto.INT64: np.int64,
        TensorProto.INT32: np.int32,
        TensorProto.INT16: np.int16,
        TensorProto.INT8: np.int8,
        TensorProto.UINT64: np.uint64,
        TensorProto.UINT32: np.uint32,
        TensorProto.UINT16: np.uint16,
        TensorProto.UINT8: np.uint8,
        TensorProto.BOOL: np.bool_,
        TensorProto.FLOAT16: np.float16,
    }
    if dtype not in mapping:
        raise ValueError("unsupported ONNX dtype in shape evaluator: {}".format(dtype))
    return mapping[dtype]


def _shape_control_node_indexes(model: ModelProto) -> set:
    """Return the reverse-reachable metadata subgraph used by dynamic attrs."""
    producer = {
        output: index
        for index, node in enumerate(model.graph.node)
        for output in node.output
    }
    roots = []
    for node in model.graph.node:
        indexes = {"Reshape": [1], "Expand": [1], "Slice": [1, 2, 3, 4]}.get(
            node.op_type, []
        )
        roots.extend(node.input[index] for index in indexes if _has_input(node, index))
    result, pending = set(), list(roots)
    while pending:
        value = pending.pop()
        index = producer.get(value)
        if index is None or index in result:
            continue
        node = model.graph.node[index]
        if node.op_type not in _SHAPE_EVALUATOR_OPS:
            continue
        result.add(index)
        # Shape consumes dimensions, not its input's activation-producing graph.
        if node.op_type != "Shape":
            pending.extend(name for name in node.input if name)
    return result
