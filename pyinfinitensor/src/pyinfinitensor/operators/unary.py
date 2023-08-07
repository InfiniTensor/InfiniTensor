import backend
from utils import _parse_attribute
from onnx import NodeProto


def import_unary(
    node: NodeProto,
    opset: int,
    tensors: dict[str, backend.Tensor],
    handler: backend.GraphHandler,
):
    # Abs
    # if opset < 6:  supported types
    # if opset < 13: supported types
    # Ceil
    # if opset < 6:  legacy optimization attribute
    # if opset < 13: supported types
    # Erf
    # if opset < 13: supported types
    # Exp
    # if opset < 6:  legacy optimization attribute
    # if opset < 13: supported types
    # Floor
    # if opset < 6:  legacy optimization attribute
    # if opset < 13: supported types
    # Log
    # if opset < 6:  legacy optimization attribute
    # if opset < 13: supported types
    # Neg
    # if opset < 6:  legacy optimization attribute
    # if opset < 13: supported types
    # Relu
    # if opset < 6:  legacy optimization attribute
    # if opset < 13: supported types
    # if opset < 14: supported types
    # Sigmoid
    # if opset < 6:  legacy optimization attribute
    # if opset < 13: supported types
    # Sqrt
    # if opset < 6:  legacy optimization attribute
    # if opset < 13: supported types
    # Tanh
    # if opset < 6:  legacy optimization attribute
    # if opset < 13: supported types
    tensors[node.output[0]] = handler.unary(
        node.op_type,
        tensors[node.input[0]],
        tensors.get(node.output[0]),
    )
