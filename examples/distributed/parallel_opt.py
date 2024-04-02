import onnx
from onnx import ModelProto, NodeProto, TensorProto, ValueInfoProto
from onnx import helper, numpy_helper
from typing import Dict, List
from placement import Placement, Replicate, Shard, _Partial
import numpy as np


def parallel_model(model: ModelProto, tp_world_size: int = 1, tp_rank: int = 0):
    data = {init.name: init for init in model.graph.initializer}
    vinfo = {info.name: info for info in model.graph.value_info}
    vinfo.update({info.name: info for info in model.graph.input})
    vinfo.update({info.name: info for info in model.graph.output})
    output = {info.name: info for info in model.graph.output}
    place: Dict[str, Placement] = {}
    nodes: List[NodeProto] = []

    def is_sharded(name: str):
        return place[name].is_shard()

    def shard_tensor(tensor: TensorProto, plc: Shard, groups: int = 1):
        # print(f"shard {tensor.name} at dim {dim}")
        assert plc.is_shard(), plc
        ndim = len(tensor.dims)
        if plc.dim < 0:
            plc.dim += ndim
        if tensor.dims[plc.dim] == 1:  # broadcast dim, no need to shard.
            return tensor
        array = numpy_helper.to_array(tensor)
        assert array.shape[plc.dim] % tp_world_size == 0, array.shape[plc.dim]
        dims = list(tensor.dims)
        dims.insert(plc.dim, groups)
        dims[plc.dim + 1] //= groups
        array = array.reshape(dims)
        seg = array.shape[plc.dim + 1] // tp_world_size
        array = array.take(
            indices=range(tp_rank * seg, (tp_rank + 1) * seg), axis=plc.dim + 1
        )
        dims = list(tensor.dims)
        dims[plc.dim] //= tp_world_size
        array = array.reshape(dims)
        tensor = numpy_helper.from_array(array, name=tensor.name)
        place[tensor.name] = plc
        return tensor

    def shard_gemm(node: NodeProto, groups: int = 1):
        # print("gemm", node.name)
        in_plc = place[node.input[0]]
        w_plc = Shard(-1) if in_plc.is_replicate() else Shard(0)
        transB = next((attr.i for attr in node.attribute if attr.name == "transB"), 0)
        if transB:
            w_plc.dim = ~w_plc.dim
        input = node.input[1]
        data[input] = shard_tensor(data[input], w_plc, groups)

        output = node.output[0]
        ndim = len(vinfo[output].type.tensor_type.shape.dim)
        out_plc = Shard(ndim - 1) if in_plc.is_replicate() else _Partial()
        place[node.output[0]] = out_plc

    def shard_concat(node: NodeProto):
        # hack for kvcache
        in_plc = place[node.input[1]]
        if in_plc.is_shard():
            seq_len_dim = vinfo[node.input[0]].type.tensor_type.shape.dim.pop(1)
            seq_len_dim.dim_value //= tp_world_size
            vinfo[node.input[0]].type.tensor_type.shape.dim.insert(1, seq_len_dim)
            place[node.input[0]] = in_plc
            place[node.output[0]] = in_plc

    def shard_binary(node: NodeProto, groups: int = 1):
        # print("binary", node.name, node.input[0], place[node.input[0]])
        a = node.input[0]
        b = node.input[1]
        if a in data:
            a, b = b, a
        place[node.output[0]] = place[a]
        if is_sharded(a) and b in data and len(data[b].dims) == 1:  # broadcast
            data[b] = shard_tensor(data[b], Shard(0), groups)

    def shard_reshape(node: NodeProto):
        # print("reshape", node.name, node.input[0], place[node.input[0]])
        # import pdb; pdb.set_trace()
        if not is_sharded(node.input[0]):
            return
        in_plc = place[node.input[0]]
        s_dim = -1
        in_dims = [d.dim_value for d in vinfo[node.input[0]].type.tensor_type.shape.dim]
        tensor = data[node.input[1]]
        out_dims = numpy_helper.to_array(tensor).copy()
        if len(in_dims) == 3 and len(out_dims) == 4:
            if in_plc.dim == 0:
                s_dim = 1
            elif in_plc.dim == 2:
                s_dim = 2
        if len(in_dims) == 4 and len(out_dims) == 3:
            if in_plc.dim == 1:
                s_dim = 0
            elif in_plc.dim == 2:
                s_dim = 2
        if len(in_dims) == 2 and len(out_dims) == 3:
            if in_plc.dim == 1:
                s_dim = 2
        if len(in_dims) == 4 and len(out_dims) == 2:
            if in_plc.dim == 1:
                s_dim = 0
            elif in_plc.dim == 2:
                s_dim = 1
        if len(in_dims) == 3 and len(out_dims) == 2:
            if in_plc.dim == 1:
                s_dim = 0
            elif in_plc.dim == 2:
                s_dim = 1
        # import pdb; pdb.set_trace()
        assert s_dim != -1
        assert out_dims[s_dim] % tp_world_size == 0, out_dims
        out_dims[s_dim] //= tp_world_size
        # if ONNX uses the same tensor for multiple Reshape Nodes, then rename it to distingush from others.
        node.input[1] = node.output[0] + "_shape"
        data[node.input[1]] = numpy_helper.from_array(out_dims, name=node.input[1])
        place[node.output[0]] = Shard(s_dim)

    def shard_split(node: NodeProto):
        if not is_sharded(node.input[0]):
            return
        in_plc = place[node.input[0]]
        split_tensor = data[node.input[1]]
        split = numpy_helper.to_array(split_tensor).copy()
        split //= tp_world_size
        data[node.input[1]] = numpy_helper.from_array(split, name=node.input[1])
        for output in node.output:
            place[output] = in_plc

    def shard_transpose(node: NodeProto):
        plc = place[node.input[0]]
        if plc.is_shard():
            perm = next(attr.ints for attr in node.attribute if attr.name == "perm")
            place[node.output[0]] = Shard(list(perm).index(plc.dim))

    def shard_node(node: NodeProto):
        if node.op_type in ["Relu", "Tanh", "Softmax", "Cast"]:
            place[node.output[0]] = place[node.input[0]]
        elif node.op_type in ["Where"]:
            place[node.output[0]] = place[node.input[1]]
        if node.op_type in {"Add", "Mul", "Div", "Max"}:
            shard_binary(node)
        elif node.op_type == "Reshape":
            shard_reshape(node)
        elif node.op_type == "Transpose":
            shard_transpose(node)
        elif node.op_type == "Split":
            shard_split(node)
        elif node.op_type == "MatMul":
            assert (
                place[node.input[0]] == place[node.input[1]]
            ), f"{place[node.input[0]]} != {place[node.input[1]]}"
            place[node.output[0]] = place[node.input[0]]
        elif node.op_type == "Concat":
            shard_concat(node)

    def find_successor(op_type: str, idx: int, search_limit: int = 1):
        for node in model.graph.node[idx + 1 : idx + 1 + search_limit]:
            if node.op_type == op_type:
                return node
        return None

    # all tensors are initially replicated.
    for v in vinfo:
        place[v] = Replicate()

    for t in data:
        place[t] = Replicate()

    for index, node in enumerate(model.graph.node):
        nodes.append(node)
        # linear
        if (node.op_type == "MatMul" or node.op_type == "Gemm") and any(
            input in data for input in node.input
        ):
            # FIXME(constroy): the last MatMul should not be sharded as TP.
            if (
                node.output[0] in output
                or (
                    index + 1 < len(model.graph.node)
                    and model.graph.node[index + 1].output[0]
                )
                in output
            ):
                continue
            groups = 1
            # If the Gemm or Matmul is followed by a split, then the inputs are concatinated by groups
            split_node = find_successor("Split", index, search_limit=2)
            if split_node is not None:
                groups = len(split_node.output)
            shard_gemm(node, groups)
            plc = place[node.output[0]]
            if plc.is_partial():
                new_name = node.output[0] + f":{plc}"
                place[new_name] = place[node.output[0]]
                # insert all_reduce
                nodes.append(
                    helper.make_node(
                        op_type="ReduceSum",
                        inputs=[new_name],
                        outputs=[node.output[0]],
                        name=node.name + "/all_reduce",
                        noop_with_empty_axes=1,
                        communicator=0,  # hack to treat ReduceSum as AllReduceSum
                    )
                )
                place[node.output[0]] = Replicate()
                node.output[0] = new_name
            if len(node.input) > 2:  # split bias to add
                prev = nodes[-1]
                new_name = prev.output[0] + "_no_bias"
                place[new_name] = place[node.output[0]]
                bias = helper.make_node(
                    op_type="Add",
                    inputs=[new_name, node.input[2]],
                    outputs=[prev.output[0]],
                    name=node.name + "/bias",
                )
                node.input.pop()
                prev.output[0] = new_name
                shard_binary(bias, groups)
                nodes.append(bias)
            continue
        shard_node(node)

    new_input = []
    for info in model.graph.input:
        new_input.append(vinfo[info.name])

    graph = helper.make_graph(
        nodes,
        model.graph.name + f"_{tp_rank}",
        new_input,
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

if __name__ == "__main__":
    model = onnx.load("./models/gpt2/gpt2_1_100.onnx")
    models = parallel_model(model, 2, 0)