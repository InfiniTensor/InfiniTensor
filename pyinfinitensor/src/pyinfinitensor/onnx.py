import backend
from onnx import ModelProto, NodeProto, TensorProto, AttributeProto, numpy_helper
from backend import DimExpr, refactor_tensor
from typing import Any


def build_graph(model: ModelProto):
    nodes: dict[str, backend.Node] = dict()
    edges: dict[str, backend.Edge] = dict()
    topology: dict[str, tuple[list[str], list[str]]] = dict()

    for tensor in model.graph.initializer:
        edges[tensor.name] = _parse_tensor(tensor)

    for tensor in model.graph.input:
        if tensor.name not in edges:
            dim = [
                DimExpr(d.dim_value) if d.dim_value > 0 else DimExpr(d.dim_param)
                for d in tensor.type.tensor_type.shape.dim
            ]
            edges[tensor.name] = refactor_tensor(
                tensor.type.tensor_type.elem_type, dim, None
            )

    for node in model.graph.node:
        topology[node.name] = ([i for i in node.input], [o for o in node.output])
        nodes[node.name] = backend.refactor_operator(
            node.op_type, _parse_attribute(node)
        )

    graph = backend.refactor_graph(
        topology,
        nodes,
        edges,
        [i.name for i in model.graph.input],
        [o.name for o in model.graph.output],
    )


def _parse_tensor(tensor: TensorProto) -> backend.Tensor:
    refactor_tensor(
        tensor.data_type,
        [DimExpr(d) for d in tensor.dims],
        [b for b in numpy_helper.to_array(tensor).data.tobytes()],
    )


def _parse_attribute(node: NodeProto) -> dict[str, Any]:
    ans: dict[str, Any] = dict()
    for attr in node.attribute:
        if attr.type == AttributeProto.INT:
            ans[attr.name] = attr.i
        elif attr.type == AttributeProto.INTS:
            ans[attr.name] = attr.ints
        elif attr.type == AttributeProto.FLOAT:
            ans[attr.name] = attr.f
        elif attr.type == AttributeProto.FLOATS:
            ans[attr.name] = attr.floats
        elif attr.type == AttributeProto.STRING:
            ans[attr.name] = attr.s
        elif attr.type == AttributeProto.STRINGS:
            ans[attr.name] = attr.strings
        elif attr.type == AttributeProto.TENSOR:
            ans[attr.name] = _parse_tensor(attr.t)
        elif attr.type == AttributeProto.TENSORS:
            ans[attr.name] = [_parse_tensor(t) for t in attr.tensors]
        else:
            assert False, "Unsupported Attribute Type: {}".format(attr.type)
    return ans
