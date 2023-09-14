import backend
from onnx import ModelProto, NodeProto, TensorProto, AttributeProto, numpy_helper
from backend import DimExpr, refactor_tensor, refactor_operator, refactor_graph
from typing import Any


def build_graph(model: ModelProto) -> backend.Graph:
    edges: dict[str, backend.Tensor] = dict()

    for tensor in model.graph.initializer:
        edges[tensor.name] = _parse_tensor(tensor)

    for tensor in model.graph.input:
        if tensor.name not in edges:
            edges[tensor.name] = refactor_tensor(
                tensor.type.tensor_type.elem_type,
                [
                    DimExpr(d.dim_value)
                    if d.HasField("dim_value")
                    else DimExpr(d.dim_param)
                    for d in tensor.type.tensor_type.shape.dim
                ],
                None,
            )

    return refactor_graph(
        {node.name: (node.input, node.output) for node in model.graph.node},
        {
            node.name: refactor_operator(node.op_type, _parse_attribute(node))
            for node in model.graph.node
        },
        edges,
        [i.name for i in model.graph.input],
        [o.name for o in model.graph.output],
    )


def _parse_tensor(tensor: TensorProto) -> backend.Tensor:
    return refactor_tensor(
        tensor.data_type,
        [DimExpr(d) for d in tensor.dims],
        numpy_helper.to_array(tensor),
    )


def _raise(attr: AttributeProto) -> None:
    raise NotImplementedError("Unsupported Attribute Type: {}".format(attr.type))


def _parse_attribute(node: NodeProto) -> dict[str, Any]:
    return {
        attr.name: attr.i
        if attr.type == AttributeProto.INT
        else attr.ints
        if attr.type == AttributeProto.INTS
        else attr.f
        if attr.type == AttributeProto.FLOAT
        else attr.floats
        if attr.type == AttributeProto.FLOATS
        else attr.s
        if attr.type == AttributeProto.STRING
        else attr.strings
        if attr.type == AttributeProto.STRINGS
        else _parse_tensor(attr.t)
        if attr.type == AttributeProto.TENSOR
        else [_parse_tensor(t) for t in attr.tensors]
        if attr.type == AttributeProto.TENSORS
        else _raise(attr)
        for attr in node.attribute
    }
