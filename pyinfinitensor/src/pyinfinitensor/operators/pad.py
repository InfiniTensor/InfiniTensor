import backend
from utils import _parse_attribute
from onnx import NodeProto


def import_pad(
    node: NodeProto,
    opset: int,
    tensors: dict[str, backend.Tensor],
    handler: backend.GraphHandler,
):
    raise NotImplementedError
