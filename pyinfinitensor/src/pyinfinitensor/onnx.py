import backend
from onnx import ModelProto, TensorProto, NodeProto, AttributeProto, TensorShapeProto
from onnx.helper import make_node
from onnx.shape_inference import infer_shapes
from typing import Dict, List, Any
from functools import reduce

runtime = backend.cpu_runtime()


def from_onnx(model: ModelProto) -> backend.GraphHandler:
    model = infer_shapes(model)
    handler = backend.GraphHandler(runtime)

    tensors: Dict[str, backend.Tensor] = dict()
    data: Dict[str, TensorProto] = dict()

    for input in model.graph.input:
        dims = _take_shape_dim(input.type.tensor_type.shape)
        tensors[input.name] = handler.tensor(dims, input.type.tensor_type.elem_type)

    for output in model.graph.output:
        dims = _take_shape_dim(output.type.tensor_type.shape)
        tensors[output.name] = handler.tensor(dims, output.type.tensor_type.elem_type)

    for initializer in model.graph.initializer:
        data[initializer.name] = initializer

    for node in model.graph.node:
        if node.op_type == "Conv":
            attributes = _parse_attribute(
                node,
                {
                    "dilations": [1, 1],
                    "pads": [0, 0],
                    "strides": [1, 1],
                },
            )
            (d, p, s) = (attributes[name] for name in ["dilations", "pads", "strides"])
            tensors[node.output[0]] = handler.conv(
                tensors[node.input[0]],
                tensors[node.input[1]],
                tensors.get(node.output[0]),
                p[0],
                p[1],
                s[0],
                s[1],
                d[0],
                d[1],
            )
        elif node.op_type == "MatMul":
            tensors[node.output[0]] = handler.matmul(
                tensors[node.input[0]],
                tensors[node.input[1]],
                tensors.get(node.output[0]),
                False,
                False,
                None,
                backend.ActType.Linear,
            )
        elif node.op_type == "Gemm":
            attributes = _parse_attribute(
                node, {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0}
            )
            (alpha, beta, transA, transB) = (
                attributes[name] for name in ["alpha", "beta", "transA", "transB"]
            )
            # FIXME 不支持 `alpha` `beta`
            assert alpha == 1.0
            assert beta == 1.0
            tensors[node.output[0]] = handler.matmul(
                tensors[node.input[0]],
                tensors[node.input[1]],
                tensors.get(node.output[0]),
                transA == 1,
                transB == 1,
                tensors[node.input[2]] if len(node.input) > 2 else None,
                backend.ActType.Linear,
            )
        elif node.op_type == "BatchNormalization":
            (input, mean, var, scale, bias) = (
                tensors[node.input[i]] for i in [0, 3, 4, 1, 2]
            )
            output = tensors.get(node.output[0])
            attributes = _parse_attribute(
                node, {"momentum": 0.9, "epsilon": 1e-05, "training_mode": 0}
            )
            (momentum, eps, training) = (
                attributes[name] for name in ["momentum", "epsilon", "training_mode"]
            )
            tensors[node.output[0]] = handler.batchNorm(
                input, output, mean, var, scale, bias, momentum, eps, training != 0
            )
        elif node.op_type == "MaxPool":
            attributes = _parse_attribute(
                node,
                {
                    "kernel_shape": None,
                    "dilations": [1, 1],
                    "pads": [0, 0],
                    "strides": [1, 1],
                },
            )
            (k, d, p, s) = (
                attributes[name]
                for name in ["kernel_shape", "dilations", "pads", "strides"]
            )
            tensors[node.output[0]] = handler.maxPool(
                tensors[node.input[0]],
                tensors.get(node.output[0]),
                k[0],
                k[1],
                d[0],
                d[1],
                p[0],
                p[1],
                s[0],
                s[1],
            )
        elif node.op_type == "AveragePool":
            attributes = _parse_attribute(
                node,
                {
                    "kernel_shape": None,
                    "pads": [0, 0],
                    "strides": [1, 1],
                },
            )
            (k, p, s) = (
                attributes[name] for name in ["kernel_shape", "pads", "strides"]
            )
            tensors[node.output[0]] = handler.avgPool(
                tensors[node.input[0]],
                tensors.get(node.output[0]),
                k[0],
                k[1],
                1,
                1,
                p[0],
                p[1],
                s[0],
                s[1],
            )
        elif node.op_type == "GlobalAveragePool":
            shape = next(
                (
                    value.type.tensor_type.shape
                    for value in model.graph.value_info
                    if value.name == node.input[0]
                ),
                None,
            ) or next(
                input.type.tensor_type.shape
                for input in model.graph.input
                if input.name == node.input[0]
            )
            [_, _, h, w] = _take_shape_dim(shape)
            tensors[node.output[0]] = handler.avgPool(
                tensors[node.input[0]],
                tensors.get(node.output[0]),
                h,
                w,
                1,
                1,
                0,
                0,
                1,
                1,
            )
        elif node.op_type == "Add":
            tensors[node.output[0]] = handler.add(
                tensors[node.input[0]],
                tensors[node.input[1]],
                tensors.get(node.output[0]),
            )
        elif node.op_type == "Sub":
            tensors[node.output[0]] = handler.sub(
                tensors[node.input[0]],
                tensors[node.input[1]],
                tensors.get(node.output[0]),
            )
        elif node.op_type == "Mul":
            tensors[node.output[0]] = handler.mul(
                tensors[node.input[0]],
                tensors[node.input[1]],
                tensors.get(node.output[0]),
            )
        elif node.op_type == "Div":
            tensors[node.output[0]] = handler.div(
                tensors[node.input[0]],
                tensors[node.input[1]],
                tensors.get(node.output[0]),
            )
        elif node.op_type == "Pow":
            tensors[node.output[0]] = handler.pow(
                tensors[node.input[0]],
                tensors[node.input[1]],
                tensors.get(node.output[0]),
            )
        elif node.op_type == "Relu":
            tensors[node.output[0]] = handler.relu(
                tensors[node.input[0]],
                tensors.get(node.output[0]),
            )
        elif node.op_type == "Sigmoid":
            tensors[node.output[0]] = handler.sigmoid(
                tensors[node.input[0]],
                tensors.get(node.output[0]),
            )
        elif node.op_type == "Tanh":
            tensors[node.output[0]] = handler.tanh(
                tensors[node.input[0]],
                tensors.get(node.output[0]),
            )
        elif node.op_type == "Softmax":
            tensors[node.output[0]] = handler.softmax(
                tensors[node.input[0]],
                tensors.get(node.output[0]),
            )
        elif node.op_type == "Abs":
            tensors[node.output[0]] = handler.abs(
                tensors[node.input[0]],
                tensors.get(node.output[0]),
            )
        elif node.op_type == "Identity":
            tensors[node.output[0]] = handler.identity(
                tensors[node.input[0]],
                tensors.get(node.output[0]),
            )
        elif node.op_type == "Flatten":
            # FIXME 后端算子不支持沿任意轴展开
            axis = next(
                (attr.i for attr in node.attribute if attr.name == "axis"), None
            )
            assert axis == None or axis == 1
            tensors[node.output[0]] = handler.flatten(
                tensors[node.input[0]],
                tensors.get(node.output[0]),
            )
        elif node.op_type == "Reshape":
            input_shape = next(
                (
                    value.type.tensor_type.shape
                    for value in model.graph.value_info
                    if value.name == node.input[0]
                ),
                None,
            ) or next(
                input.type.tensor_type.shape
                for input in model.graph.input
                if input.name == node.input[0]
            )
            dims = _take_shape_dim(input_shape)
            size = reduce(lambda acc, x: acc * x, dims)
            output_shape = [int(i) for i in data[node.input[1]].int64_data]
            for i, x in enumerate(output_shape):
                if x == 0:
                    output_shape[i] = dims[i]
            temp = reduce(lambda acc, x: acc * x, output_shape)
            if temp < 0:
                output_shape[output_shape.index(-1)] = size // -temp
            tensors[node.output[0]] = handler.reshape(
                tensors[node.input[0]],
                tensors.get(node.output[0]),
                output_shape,
            )
        elif node.op_type == "Concat":
            tensors[node.output[0]] = handler.concat(
                [tensors[name] for name in node.input],
                tensors.get(node.output[0]),
                next((attr.i for attr in node.attribute if attr.name == "axis")),
            )
        elif node.op_type == "Gather":
            tensors[node.output[0]] = handler.gather(
                tensors[node.input[0]],
                tensors[node.input[1]],
                tensors.get(node.output[0]),
                next((attr.i for attr in node.attribute if attr.name == "axis")),
            )
        elif node.op_type == "ReduceMean":
            tensors[node.output[0]] = handler.reduceMean(
                tensors[node.input[0]],
                tensors.get(node.output[0]),
                tensors[node.input[1]] if len(node.input) > 1 else None,
                next((attr.i for attr in node.attribute if attr.name == "keepdims"))
                != 0,
            )
        elif node.op_type == "Slice":
            tensors[node.output[0]] = handler.slice(
                tensors[node.input[0]],
                tensors.get(node.output[0]),
                _parse_data(data[node.input[1]]),
                _parse_data(data[node.input[2]]),
                _parse_data(data[node.input[3]]) if len(node.input) > 3 else None,
                _parse_data(data[node.input[4]]) if len(node.input) > 4 else None,
            )
        elif node.op_type == "Pad":
            tensors[node.output[0]] = handler.pad(
                tensors[node.input[0]],
                tensors.get(node.output[0]),
                _parse_data(data[node.input[1]]),
                _parse_data(data[node.input[3]]) if len(node.input) > 3 else None,
            )
        else:
            raise Exception('Unsupported operator "{}"'.format(node.op_type))


def to_onnx(graph: backend.GraphHandler):
    if not graph.topo_sort():
        raise Exception("Sorting fails")

    ops = graph.operators()

    names: Dict[Any, str] = dict()
    nodes: List[NodeProto] = []
    count_op: Dict[backend.OpType, int] = dict()
    count_in = 0

    for op in ops:
        ty = op.op_type()
        name = "{}{}".format(ty.name, count_op.setdefault(ty, 0) + 1)
        inputs = op.inputs()
        outputs = op.outputs()
        names[op] = name
        count_op[ty] += 1
        if ty == backend.OpType.Matmul:
            raise Exception("TODO")
        elif ty == backend.OpType.BatchNorm:
            raise Exception("TODO")
        elif ty == backend.OpType.MaxPool:
            raise Exception("TODO")
        elif ty == backend.OpType.AvgPool:
            raise Exception("TODO")
        elif ty == backend.OpType.Add:
            names[outputs[0]] = name
            if inputs[0] in names:
                a = names[inputs[0]]
            else:
                count_in += 1
                a = "input{}".format(count_in)
            if inputs[1] in names:
                b = names[inputs[1]]
            else:
                count_in += 1
                b = "input{}".format(count_in)
            nodes.append(make_node("Add", [a, b], [name], name))
        elif ty == backend.OpType.Sub:
            names[outputs[0]] = name
            if inputs[0] in names:
                a = names[inputs[0]]
            else:
                count_in += 1
                a = "input{}".format(count_in)
            if inputs[1] in names:
                b = names[inputs[1]]
            else:
                count_in += 1
                b = "input{}".format(count_in)
            nodes.append(make_node("Sub", [a, b], [name], name))
        elif ty == backend.OpType.Mul:
            names[outputs[0]] = name
            if inputs[0] in names:
                a = names[inputs[0]]
            else:
                count_in += 1
                a = "input{}".format(count_in)
            if inputs[1] in names:
                b = names[inputs[1]]
            else:
                count_in += 1
                b = "input{}".format(count_in)
            nodes.append(make_node("Mul", [a, b], [name], name))
        elif ty == backend.OpType.Div:
            names[outputs[0]] = name
            if inputs[0] in names:
                a = names[inputs[0]]
            else:
                count_in += 1
                a = "input{}".format(count_in)
            if inputs[1] in names:
                b = names[inputs[1]]
            else:
                count_in += 1
                b = "input{}".format(count_in)
            nodes.append(make_node("Div", [a, b], [name], name))
        elif ty == backend.OpType.Pow:
            names[outputs[0]] = name
            if inputs[0] in names:
                a = names[inputs[0]]
            else:
                count_in += 1
                a = "input{}".format(count_in)
            if inputs[1] in names:
                b = names[inputs[1]]
            else:
                count_in += 1
                b = "input{}".format(count_in)
            nodes.append(make_node("Pow", [a, b], [name], name))
        elif ty == backend.OpType.Relu:
            raise Exception("TODO")
        elif ty == backend.OpType.Sigmoid:
            raise Exception("TODO")
        elif ty == backend.OpType.Tanh:
            raise Exception("TODO")
        elif ty == backend.OpType.Softmax:
            raise Exception("TODO")
        elif ty == backend.OpType.Abs:
            raise Exception("TODO")
        elif ty == backend.OpType.Identity:
            raise Exception("TODO")
        elif ty == backend.OpType.Flatten:
            raise Exception("TODO")
        elif ty == backend.OpType.Reshape:
            raise Exception("TODO")
        elif ty == backend.OpType.Concat:
            raise Exception("TODO")
        elif ty == backend.OpType.Gather:
            raise Exception("TODO")
        elif ty == backend.OpType.ReduceMean:
            raise Exception("TODO")
        elif ty == backend.OpType.Slice:
            raise Exception("TODO")
        elif ty == backend.OpType.Pad:
            raise Exception("TODO")
        else:
            raise Exception("Unsupported OpType {}".format(ty.name))

    print(names)


def parse_onnx(model: ModelProto):
    print()

    for field in [
        "doc_string",
        "domain",
        "functions",
        "metadata_props",
        "model_version",
        "producer_name",
        "producer_version",
        "training_info",
    ]:
        print("{}: {}".format(field, getattr(model, field)))

    print("ir_version:", model.ir_version)
    for opset in model.opset_import:
        print("opset domain={} version={}".format(opset.domain, opset.version))

    print("layout:")
    for node in model.graph.node:
        print(
            '   {o} <- {op}"{name}"{a} <- {i}'.format(
                name=node.name,
                op=node.op_type,
                i=node.input,
                o=node.output,
                a=[a.name for a in node.attribute],
            )
        )

    print("weight:")
    for node in model.graph.initializer:
        print("   {}".format(node.name))


def _parse_attribute(node: NodeProto, attrs: Dict[str, Any] = dict()) -> Dict[str, Any]:
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


def _parse_data(tensor: TensorProto) -> List[int]:
    if tensor.data_type == TensorProto.INT32:
        return [int(i) for i in tensor.int32_data]
    elif tensor.data_type == TensorProto.INT64:
        return [int(i) for i in tensor.int64_data]
    else:
        assert False, "Unsupported Tensor Type: {}".format(tensor.data_type)


def _take_shape_dim(shape: TensorShapeProto) -> List[int]:
    return [(d.dim_value if d.dim_value > 0 else 1) for d in shape.dim]
