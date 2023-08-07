import backend
from utils import _parse_attribute
from onnx import NodeProto


def import_binary(
    node: NodeProto,
    opset: int,
    tensors: dict[str, backend.Tensor],
    handler: backend.GraphHandler,
):
    # Add
    # if opset < 6:  lagacy optimization attribute
    # if opset < 7:  not default broadcastable
    # if opset < 13: supported types
    # if opset < 14: supported types
    # Sub
    # if opset < 6:  lagacy optimization attribute
    # if opset < 7:  not default broadcastable
    # if opset < 13: supported types
    # if opset < 14: supported types
    # Mul
    # if opset < 6:  lagacy optimization attribute
    # if opset < 7:  not default broadcastable
    # if opset < 13: supported types
    # if opset < 14: supported types
    # Div
    # if opset < 6:  lagacy optimization attribute
    # if opset < 7:  not default broadcastable
    # if opset < 13: supported types
    # if opset < 14: supported types
    # Pow
    # if opset < 7:  not default broadcastable
    # if opset < 12: supported types
    # if opset < 13: supported types
    # if opset < 15: supported types
    # And
    # if opset < 7:  not default broadcastable
    # Or
    # if opset < 7:  not default broadcastable
    # Xor
    # if opset < 7:  not default broadcastable
    # Equal
    # if opset < 7:  not default broadcastable
    # if opset < 11: supported types
    # if opset < 13: supported types
    # if opset < 19: supported types
    if (
        node.op_type in ["Add", "Sub", "Mul", "Div", "Pow", "And", "Or", "Xor", "Equal"]
        and opset < 7
    ):
        raise ValueError("Need special handling for non-default broadcastable")
    tensors[node.output[0]] = handler.binary(
        node.op_type,
        tensors[node.input[0]],
        tensors[node.input[1]],
        tensors.get(node.output[0]),
    )
