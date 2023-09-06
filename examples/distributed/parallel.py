import onnx
from onnx import (
    ModelProto,
    TensorProto,
    NodeProto,
    AttributeProto,
)
from onnx import helper, numpy_helper
from typing import Dict, Any


def parse_attribute(node: NodeProto, attrs: Dict[str, Any] = dict()) -> Dict[str, Any]:
    for attr in node.attribute:
        if attr.name in attrs:
            if attr.type == AttributeProto.INT:
                attrs[attr.name] = attr.i
            elif attr.type == AttributeProto.INTS:
                attrs[attr.name] = attr.ints
            elif attr.type == AttributeProto.FLOAT:
                attrs[attr.name] = attr.f
            elif attr.type == AttributeProto.STRING:
                attrs[attr.name] = attr.s
            elif attr.type == AttributeProto.TENSOR:
                attrs[attr.name] = attr.t
            else:
                assert False, "Unsupported Attribute Type: {}".format(attr.type)
    return attrs


def parallel_model(model: ModelProto, tp_world_size: int = 1, tp_rank: int = 0):
    data = {init.name: init for init in model.graph.initializer}
    nodes = list(model.graph.node)

    def shard_tensor(tensor: TensorProto, dim: int):
        array = numpy_helper.to_array(tensor)
        if dim >= array.ndim:
            dim = array.ndim - 1
        assert array.shape[dim] % tp_world_size == 0
        seg = array.shape[dim] // tp_world_size
        array = array[tp_rank * seg : (tp_rank + 1) * seg]
        return numpy_helper.from_array(array, name=tensor.name + f":sharded({dim})")

    def shard_gemm(node: NodeProto):
        attrs = parse_attribute(
            node, {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0}
        )
        trans = [attrs["transA"], attrs["transB"]]
        dim = 0
        for i, (input, t) in enumerate(zip(node.input, trans)):
            if input in data:
                dim = i
                sharded = shard_tensor(data[input], dim ^ t)
                node.input[i] = sharded.name
                data[input] = sharded
        if len(node.input) > 2:
            input = node.input[2]
            sharded = shard_tensor(data[input], dim)
            node.input[2] = sharded.name
            data[input] = sharded

        node.output[0] += f":sharded({dim})"
        return dim

    for i, node in enumerate(nodes):
        if node.op_type == "Gemm":
            output = node.output[0]
            dim = shard_gemm(node)
            gathered = [node.output[0] + f".{i}" for i in range(tp_world_size)]
            # all_gather
            nodes.insert(
                i + 1,
                helper.make_node(
                    op_type="AllGather",
                    inputs=[node.output[0]],
                    outputs=gathered,
                    name=node.name + "/allgather",
                    # domain="infini", # shape inference fails for custom domain
                ),
            )
            # concat
            nodes.insert(
                i + 2,
                helper.make_node(
                    op_type="Concat",
                    inputs=gathered,
                    outputs=[output],
                    name=node.name + "/concat",
                    axis=dim,
                ),
            )
    graph = helper.make_graph(
        nodes,
        model.graph.name + f"_{tp_rank}",
        model.graph.input,
        model.graph.output,
        data.values(),
        doc_string=model.graph.doc_string,
        value_info=model.graph.value_info,
    )
    model = helper.make_model(graph)

    onnx.shape_inference.infer_shapes(model)
    return model
