import backend
from utils import _parse_attribute
from onnx import NodeProto


def import_matmul(
    node: NodeProto,
    opset: int,
    tensors: dict[str, backend.Tensor],
    handler: backend.GraphHandler,
):
    # if opset < 9:
    # if opset < 13:
    tensors[node.output[0]] = handler.matmul(
        tensors[node.input[0]],
        tensors[node.input[1]],
        tensors.get(node.output[0]),
        False,
        False,
        None,
        backend.ActType.Linear,
    )
