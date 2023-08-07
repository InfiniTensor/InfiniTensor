import backend
from utils import _parse_attribute
from onnx import NodeProto


def import_conv(
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
            "pads": [0, 0, 0, 0],
            "strides": [1, 1],
        },
    )
    (d, p, s) = (attributes[name] for name in ["dilations", "pads", "strides"])
    if p[0] != p[2] or p[1] != p[3]:
        adapt = "{}-adapt".format(node.output[0])
        tensors[adapt] = handler.pad(tensors[node.input[0]], None, p, [-2, -1])
        p = [0, 0, 0, 0]
    else:
        adapt = node.input[0]

    tensors[node.output[0]] = handler.conv(
        tensors[adapt],
        tensors[node.input[1]],
        tensors[node.input[2]] if len(node.input) > 2 else None,
        tensors.get(node.output[0]),
        p[0],
        p[1],
        s[0],
        s[1],
        d[0],
        d[1],
    )
