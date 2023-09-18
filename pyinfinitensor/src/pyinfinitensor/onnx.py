import backend
from onnx import ModelProto, NodeProto, TensorProto, AttributeProto, numpy_helper
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info
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

    x = set()
    for node in model.graph.node:
        x.add(node.op_type)
    print(x)

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


def build_onnx(grpah_name: str, graph: backend.Graph) -> ModelProto:
    node_export = backend.NodeExport(graph)
    edge_export = backend.EdgeExport(graph)
    nodes = []
    edges = {}

    while True:
        node = node_export.next()
        if node is None:
            break
        (name, op_type, attributes, inputs, outputs) = node
        nodes.append(make_node(op_type, inputs, outputs, name=name, **attributes))

    while True:
        edge = edge_export.next()
        if edge is None:
            break
        (name, data_type, shape) = edge
        edges[name] = make_tensor_value_info(name, data_type, shape)

    global_inputs = [
        edges.pop(name)
        if name in edges
        else make_tensor_value_info(name, TensorProto.UNDEFINED, None)
        for name in node_export.global_inputs()
    ]
    global_outputs = [
        edges.pop(name)
        if name in edges
        else make_tensor_value_info(name, TensorProto.UNDEFINED, None)
        for name in node_export.global_outputs()
    ]
    value_info = list(edges.values())

    return make_model(
        make_graph(
            nodes, grpah_name, global_inputs, global_outputs, value_info=value_info
        )
    )
