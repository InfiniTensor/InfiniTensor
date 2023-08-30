from enum import Enum
import onnx
from onnx import (
    ModelProto,
    TensorProto,
    NodeProto,
)
from onnx import helper, numpy_helper
from typing import Dict, List, Set


class Placement(Enum):
    Replicate = 0
    Shard = 1
    _Partial = 2


def parallel_model(model: ModelProto, tp_world_size: int = 1, tp_rank: int = 0):
    data = {init.name: init for init in model.graph.initializer}
    vinfo = {info.name: info for info in model.graph.value_info}
    vinfo.update({info.name: info for info in model.graph.input})
    vinfo.update({info.name: info for info in model.graph.output})
    nodes: List[NodeProto] = []
    renames: Dict[str, str] = {}

    def is_sharded(tensor: str):
        return Placement.Shard.name in tensor

    def shard_tensor(tensor: TensorProto, dim: int, name: str):
        # print(f"shard {tensor.name} at dim {dim}")
        ndim = len(tensor.dims)
        if dim < 0:
            dim += ndim
        if tensor.dims[dim] == 1:  # broadcast dim, no need to shard.
            return tensor
        array = numpy_helper.to_array(tensor)
        assert array.shape[dim] % tp_world_size == 0, array.shape[dim]
        seg = array.shape[dim] // tp_world_size
        array = array.take(indices=range(tp_rank * seg, (tp_rank + 1) * seg), axis=dim)
        return numpy_helper.from_array(array, name=name)

    def shard_gemm(node: NodeProto, plc: Placement):
        # print("gemm", node.name)
        dim = -1 if placement == Placement.Replicate else 0
        transB = next((attr.i for attr in node.attribute if attr.name == "transB"), 0)
        input = node.input[1]
        sharded = shard_tensor(data[input], dim ^ transB, input + f":{plc.name}({dim})")
        node.input[1] = sharded.name
        data[input] = sharded

        if len(node.input) > 2 and dim == -1:
            input = node.input[2]
            sharded = shard_tensor(data[input], dim, input + f":{plc.name}({dim})")
            node.input[2] = sharded.name
            data[input] = sharded

        plc = Placement.Shard if plc == Placement.Replicate else Placement._Partial
        return plc, dim

    def shard_bias(node: NodeProto, plc: Placement):
        # print("bias", node.name)
        dim = 0
        for i, input in enumerate(node.input):
            if input in data and len(data[input].dims) == 1:
                sharded = shard_tensor(data[input], dim, input + f":{plc.name}({dim})")
                node.input[i] = sharded.name
                data[input] = sharded
        return plc, dim

    def shard_reshape(node: NodeProto):
        # print("reshape", node.name, node.input[1])
        if not is_sharded(node.input[0]):
            return
        tensor = data[node.input[1]]
        dims = numpy_helper.to_array(tensor).copy()
        assert dims[-1] % tp_world_size == 0, dims
        dims[-1] = dims[-1] // tp_world_size
        node.input[1] = node.output[0] + "_shape"
        data[node.input[1]] = numpy_helper.from_array(dims, name=node.input[1])

    def shard_node(node: NodeProto, placement: Placement):
        def rename_output(node: NodeProto, idx: int, dim: int):
            new_name = node.output[idx] + f":{placement.name}({dim})"
            renames[node.output[idx]] = new_name
            vinfo[new_name] = vinfo.pop(node.output[idx])
            node.output[idx] = new_name

        if node.op_type == "Split" and is_sharded(node.input[0]):
            data.pop(node.input[1], None)
            node.input.pop()
            node.attribute.append(
                helper.make_attribute("num_outputs", len(node.output))
            )
            for i in range(len(node.output)):
                rename_output(node, i, -1)
        elif node.op_type in {"Add", "Mul", "Max"} and any(
            t not in data and is_sharded(t) for t in node.input
        ):
            placement, dim = shard_bias(node, placement)
            rename_output(node, 0, dim)
        elif node.op_type == "Reshape" and is_sharded(node.input[0]):
            shard_reshape(node)
            rename_output(node, 0, -1)
        elif node.op_type == "Transpose" and is_sharded(node.input[0]):
            rename_output(node, 0, -1)
        elif node.op_type == "MatMul" and (
            is_sharded(node.input[0]) ^ is_sharded(node.input[1])
        ):
            rename_output(node, 0, -1)

    # current placement of activitions.
    placement = Placement.Replicate
    for i, node in enumerate(model.graph.node):
        nodes.append(node)
        # rename input
        for i, input in enumerate(node.input):
            if input in renames:
                node.input[i] = renames[input]
        # shard linear
        if (node.op_type == "MatMul" or node.op_type == "Gemm") and any(
            input in data for input in node.input
        ):
            placement, dim = shard_gemm(node, placement)
            new_name = node.output[0] + f":{placement.name}({dim})"
            if placement == Placement._Partial:
                # insert all_reduce
                nodes.append(
                    helper.make_node(
                        op_type="ReduceSum",
                        inputs=[new_name],
                        outputs=[node.output[0]],
                        name=node.name + "/all_reduce",
                        communicator=0,  # hack to treat ReduceSum as AllReduceSum
                    )
                )
                placement = Placement.Replicate
            else:
                # rename output
                renames[node.output[0]] = new_name
                vinfo[new_name] = vinfo.pop(node.output[0])
            node.output[0] = new_name
            continue
        shard_node(node, placement)

    for output in model.graph.output:
        if output.name in renames:
            output.name = renames[output.name]
    graph = helper.make_graph(
        nodes,
        model.graph.name + f"_{tp_rank}",
        model.graph.input,
        model.graph.output,
        data.values(),
        doc_string=model.graph.doc_string,
        # value_info=vinfo.values(),
    )
    for output in graph.output:
        tt = output.type.tensor_type
        if tt.HasField("shape"):
            tt.ClearField("shape")
    model = helper.make_model(graph)
    model = onnx.shape_inference.infer_shapes(model)
    return model
