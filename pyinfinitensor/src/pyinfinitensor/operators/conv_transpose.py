import backend
from utils import _parse_attribute
from onnx import NodeProto


def import_conv_transpose(
    node: NodeProto,
    opset: int,
    tensors: dict[str, backend.Tensor],
    handler: backend.GraphHandler,
):
    # if opset < 11:
    attributes = _parse_attribute(
        node,
        {
            "dilations": [1, 1],
            "pads": [0, 0],
            "strides": [1, 1],
            "output_padding": [0, 0],
        },
    )
    (d, p, s, op) = (
        attributes[name] for name in ["dilations", "pads", "strides", "output_padding"]
    )
    tensors[node.output[0]] = handler.convTransposed2d(
        tensors[node.input[0]],
        tensors[node.input[1]],
        tensors.get(node.output[0]),
        p[0],
        p[1],
        s[0],
        s[1],
        d[0],
        d[1],
        op[0],
        op[1],
    )
