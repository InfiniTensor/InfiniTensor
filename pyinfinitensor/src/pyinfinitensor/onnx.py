import backend
from onnx import (
    ModelProto,
    TensorProto,
    NodeProto,
    AttributeProto,
    TensorShapeProto,
    ValueInfoProto,
)
from onnx.helper import (
    make_node,
    make_tensor_value_info,
    make_tensor,
    make_graph,
    make_model,
)
from onnx.checker import (
    check_graph,
    check_model,
    check_node,
    check_value_info,
    check_tensor,
)
from onnx.shape_inference import infer_shapes
from onnx.numpy_helper import to_array
from typing import Dict, List, Any, Tuple, Sequence, Union, Optional
from functools import reduce


class OnnxStub:
    """
    The Onnx model imported into infinitensor.
    It can be generated from an Onnx model object.
    """

    inputs: Dict[str, backend.Tensor] = {}
    outputs: Dict[str, backend.Tensor] = {}
    initializer: Dict[int, TensorProto] = {}
    handler: backend.GraphHandler

    def __init__(self, model: ModelProto, runtime):
        model = infer_shapes(model)
        self.handler = backend.GraphHandler(runtime)

        tensors: Dict[str, backend.Tensor] = dict()
        data: Dict[str, TensorProto] = dict()

        for input in model.graph.input:
            dims = _take_shape_dim(input.type.tensor_type.shape)
            tensors[input.name] = self.handler.tensor(
                dims, input.type.tensor_type.elem_type
            )

        for output in model.graph.output:
            dims = _take_shape_dim(output.type.tensor_type.shape)
            tensors[output.name] = self.handler.tensor(
                dims, output.type.tensor_type.elem_type
            )

        for initializer in model.graph.initializer:
            dims = [d for d in initializer.dims]
            tensors[initializer.name] = self.handler.tensor(dims, initializer.data_type)
            data[initializer.name] = initializer

        node_name = []
        new_node_name = []
        for node in model.graph.node:
            node_name.append(node.name)
        node_list = model.graph.node
        while len(node_list) != 0:
            for node in model.graph.node:
                if node.name not in node_list:
                    continue
                if _analyse_node(node, tensors):
                    continue
                if node.op_type == "Conv":
                    attributes = _parse_attribute(
                        node,
                        {
                            "dilations": [1, 1],
                            "pads": [0, 0, 0, 0],
                            "strides": [1, 1],
                        },
                    )
                    (d, p, s) = (
                        attributes[name] for name in ["dilations", "pads", "strides"]
                    )
                    if p[0] != p[2] or p[1] != p[3]:
                        adapt = "{}-adapt".format(node.output[0])
                        tensors[adapt] = self.handler.pad(
                            tensors[node.input[0]], None, p, [-2, -1]
                        )
                        p = [0, 0, 0, 0]
                    else:
                        adapt = node.input[0]

                    if len(node.input) > 2:
                        bias = "{}-bias".format(node.output[0])
                        reshape = "{}-reshape".format(node.output[0])
                        tensors[bias] = self.handler.conv(
                            tensors[adapt],
                            tensors[node.input[1]],
                            None,
                            p[0],
                            p[1],
                            s[0],
                            s[1],
                            d[0],
                            d[1],
                        )
                        tensors[reshape] = self.handler.reshape(
                            tensors[node.input[2]],
                            None,
                            [
                                1,
                                reduce(
                                    lambda acc, x: acc * x,
                                    _search_shape(model, node.input[2]),
                                ),
                                1,
                                1,
                            ],
                        )
                        tensors[node.output[0]] = self.handler.add(
                            tensors[bias],
                            tensors[reshape],
                            tensors.get(node.output[0]),
                        )
                    else:
                        tensors[node.output[0]] = self.handler.conv(
                            tensors[adapt],
                            tensors[node.input[1]],
                            tensors.get(node.output[0]),
                            p[0],
                            p[1],
                            s[0],
                            s[1],
                            d[0],
                            d[1],
                        )
                elif node.op_type == "ConvTranspose":
                    attributes = _parse_attribute(
                        node,
                        {
                            "dilations": [1, 1],
                            "pads": [0, 0],
                            "strides": [1, 1],
                            "output_padding": [0, 0],
                        },
                    )
                    (d, p, s, op) = (
                        attributes[name]
                        for name in ["dilations", "pads", "strides", "output_padding"]
                    )
                    tensors[node.output[0]] = self.handler.convTransposed2d(
                        tensors[node.input[0]],
                        tensors[node.input[1]],
                        tensors.get(node.output[0]),
                        p[0],
                        p[1],
                        s[0],
                        s[1],
                        d[0],
                        d[1],
                        op[0],
                        op[1],
                    )
                elif node.op_type == "MatMul":
                    tensors[node.output[0]] = self.handler.matmul(
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
                        attributes[name]
                        for name in ["alpha", "beta", "transA", "transB"]
                    )
                    # FIXME unsupport attributes: `alpha` `beta`
                    assert alpha == 1.0
                    assert beta == 1.0
                    tensors[node.output[0]] = self.handler.matmul(
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
                        attributes[name]
                        for name in ["momentum", "epsilon", "training_mode"]
                    )
                    tensors[node.output[0]] = self.handler.batchNormalization(
                        input,
                        output,
                        mean,
                        var,
                        scale,
                        bias,
                        momentum,
                        eps,
                        training != 0,
                    )
                elif node.op_type == "MaxPool":
                    attributes = _parse_attribute(
                        node,
                        {
                            "kernel_shape": None,
                            "dilations": [1, 1],
                            "pads": [0, 0, 0, 0],
                            "strides": [1, 1],
                        },
                    )
                    (k, d, p, s) = (
                        attributes[name]
                        for name in ["kernel_shape", "dilations", "pads", "strides"]
                    )
                    if p[0] != p[2] or p[1] != p[3]:
                        adapt = "{}-adapt".format(node.output[0])
                        tensors[adapt] = self.handler.pad(
                            tensors.get(node.input[0]), None, p, [-2, -1]
                        )
                        tensors[node.output[0]] = self.handler.maxPool(
                            tensors[adapt],
                            tensors.get(node.output[0]),
                            k[0],
                            k[1],
                            d[0],
                            d[1],
                            0,
                            0,
                            s[0],
                            s[1],
                        )
                    else:
                        tensors[node.output[0]] = self.handler.maxPool(
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
                            "pads": [0, 0, 0, 0],
                            "strides": [1, 1],
                        },
                    )
                    (k, p, s) = (
                        attributes[name] for name in ["kernel_shape", "pads", "strides"]
                    )
                    if p[0] != p[2] or p[1] != p[3]:
                        adapt = "{}-adapt".format(node.output[0])
                        tensors[adapt] = self.handler.pad(
                            tensors.get(node.input[0]), None, p, [-2, -1]
                        )
                        tensors[node.output[0]] = self.handler.avgPool(
                            tensors[adapt],
                            tensors.get(node.output[0]),
                            k[0],
                            k[1],
                            1,
                            1,
                            0,
                            0,
                            s[0],
                            s[1],
                        )
                    else:
                        tensors[node.output[0]] = self.handler.avgPool(
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
                    [_, _, h, w] = _search_shape(model, node.input[0])
                    tensors[node.output[0]] = self.handler.avgPool(
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
                    tensors[node.output[0]] = self.handler.add(
                        tensors[node.input[0]],
                        tensors[node.input[1]],
                        tensors.get(node.output[0]),
                    )
                elif node.op_type == "Sub":
                    tensors[node.output[0]] = self.handler.sub(
                        tensors[node.input[0]],
                        tensors[node.input[1]],
                        tensors.get(node.output[0]),
                    )
                elif node.op_type == "Mul":
                    tensors[node.output[0]] = self.handler.mul(
                        tensors[node.input[0]],
                        tensors[node.input[1]],
                        tensors.get(node.output[0]),
                    )
                elif node.op_type == "Div":
                    tensors[node.output[0]] = self.handler.div(
                        tensors[node.input[0]],
                        tensors[node.input[1]],
                        tensors.get(node.output[0]),
                    )
                elif node.op_type == "Pow":
                    tensors[node.output[0]] = self.handler.pow(
                        tensors[node.input[0]],
                        tensors[node.input[1]],
                        tensors.get(node.output[0]),
                    )
                elif node.op_type == "Min":
                    tensors[node.output[0]] = self.handler.min(
                        tensors[node.input[0]],
                        tensors[node.input[1]],
                        tensors.get(node.output[0]),
                    )
                elif node.op_type == "Max":
                    tensors[node.output[0]] = self.handler.max(
                        tensors[node.input[0]],
                        tensors[node.input[1]],
                        tensors.get(node.output[0]),
                    )
                elif node.op_type == "Relu":
                    tensors[node.output[0]] = self.handler.relu(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                    )
                elif node.op_type == "Sigmoid":
                    tensors[node.output[0]] = self.handler.sigmoid(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                    )
                elif node.op_type == "Tanh":
                    tensors[node.output[0]] = self.handler.tanh(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                    )
                elif node.op_type == "Softmax":
                    tensors[node.output[0]] = self.handler.softmax(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                        next(
                            (attr.i for attr in node.attribute if attr.name == "axis"),
                            -1,
                        ),
                    )
                elif node.op_type == "Abs":
                    tensors[node.output[0]] = self.handler.abs(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                    )
                elif node.op_type == "Sqrt":
                    tensors[node.output[0]] = self.handler.sqrt(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                    )
                elif node.op_type == "Shape":
                    tensors[node.output[0]] = self.handler.shape(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                    )
                elif node.op_type == "Identity":
                    tensors[node.output[0]] = self.handler.identity(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                    )
                elif node.op_type == "Flatten":
                    tensors[node.output[0]] = self.handler.flatten(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                        next(
                            (attr.i for attr in node.attribute if attr.name == "axis"),
                            1,
                        ),
                    )
                elif node.op_type == "PRelu":
                    tensors[node.output[0]] = self.handler.pRelu(
                        tensors[node.input[0]],
                        tensors[node.input[1]],
                        tensors.get(node.output[0]),
                    )
                elif node.op_type == "Clip":
                    tensors[node.output[0]] = self.handler.clip(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                        next(_parse_data(data[node.input[1]]).__iter__(), None)
                        if len(node.input) > 1
                        else None,
                        next(_parse_data(data[node.input[2]]).__iter__(), None)
                        if len(node.input) > 2
                        else None,
                    )
                elif node.op_type == "Transpose":
                    perm = next(
                        (attr.ints for attr in node.attribute if attr.name == "perm"),
                        None,
                    )
                    tensors[node.output[0]] = self.handler.transpose(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                        perm,
                    )
                elif node.op_type == "Reshape":
                    shape = _parse_data(data[node.input[1]])
                    tensors[node.output[0]] = self.handler.reshape(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                        shape,
                    )
                elif node.op_type == "Squeeze":
                    input_shape = _search_shape(model, node.input[0])
                    axes = set(
                        [int(i) for i in data[node.input[1]].int64_data]
                        if len(node.input) > 1
                        else _parse_attribute(node, {"axes": None})["axes"]
                    )
                    assert all(input_shape[d] == 1 for d in axes)
                    output_shape = []
                    for i, x in enumerate(input_shape):
                        if i not in axes:
                            output_shape.append(x)
                    tensors[node.output[0]] = self.handler.reshape(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                        output_shape,
                    )
                elif node.op_type == "Unsqueeze":
                    input_shape = _search_shape(model, node.input[0])
                    axes = (
                        [int(i) for i in data[node.input[1]].int64_data]
                        if len(node.input) > 1
                        else _parse_attribute(node, {"axes": None})["axes"]
                    )
                    for i in axes:
                        input_shape.insert(i, 1)
                    tensors[node.output[0]] = self.handler.reshape(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                        input_shape,
                    )
                elif node.op_type == "Concat":
                    tensors[node.output[0]] = self.handler.concat(
                        [tensors[name] for name in node.input],
                        tensors.get(node.output[0]),
                        next(
                            (attr.i for attr in node.attribute if attr.name == "axis")
                        ),
                    )
                elif node.op_type == "Split":
                    for name, tensor in zip(
                        node.output,
                        self.handler.split(
                            tensors[node.input[0]],
                            None,
                            next(
                                (
                                    attr.i
                                    for attr in node.attribute
                                    if attr.name == "axis"
                                ),
                                0,
                            ),
                            len(node.output),
                        ),
                    ):
                        tensors[name] = tensor
                elif node.op_type == "Gather":
                    tensors[node.output[0]] = self.handler.gather(
                        tensors[node.input[0]],
                        tensors[node.input[1]],
                        tensors.get(node.output[0]),
                        next(
                            (attr.i for attr in node.attribute if attr.name == "axis"),
                            0,
                        ),
                    )
                elif node.op_type == "ReduceMean":
                    tensors[node.output[0]] = self.handler.reduce_mean(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                        # NOTE(constroy): `axes` is an attribute until opset version 13.
                        next(
                            (
                                attr.ints
                                for attr in node.attribute
                                if attr.name == "axes"
                            ),
                            None,
                        ),
                        next(
                            (
                                attr.i
                                for attr in node.attribute
                                if attr.name == "keepdims"
                            ),
                            1,
                        )
                        != 0,
                    )
                elif node.op_type == "Slice":
                    tensors[node.output[0]] = self.handler.slice(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                        _parse_data(data[node.input[1]]),
                        _parse_data(data[node.input[2]]),
                        _parse_data(data[node.input[3]])
                        if len(node.input) > 3
                        else None,
                        _parse_data(data[node.input[4]])
                        if len(node.input) > 4
                        else None,
                    )
                elif node.op_type == "Pad":
                    tensors[node.output[0]] = self.handler.pad(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                        _parse_data(data[node.input[1]]),
                        _parse_data(data[node.input[3]])
                        if len(node.input) > 3
                        else None,
                    )
                elif node.op_type == "Dropout":
                    for name, tensor in zip(
                        node.output,
                        self.handler.dropout(
                            tensors[node.input[0]],
                            tensors.get(node.output[0]),
                            tensors.get(node.output[1])
                            if len(node.output) > 1
                            else None,
                            _parse_data(data[node.input[1]])[0]
                            if len(node.input) > 1
                            else 0.5,
                            _parse_data(data[node.input[2]])[0]
                            if len(node.input) > 2
                            else False,
                        ),
                    ):
                        tensors[name] = tensor
                elif node.op_type == "Cast":
                    tensors[node.output[0]] = self.handler.cast(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                        next((attr.i for attr in node.attribute if attr.name == "to")),
                    )
                elif node.op_type == "AllReduceSum":
                    tensors[node.output[0]] = self.handler.allReduceSum(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                    )
                elif node.op_type == "AllReduceProd":
                    tensors[node.output[0]] = self.handler.allReduceProd(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                    )
                elif node.op_type == "AllReduceMin":
                    tensors[node.output[0]] = self.handler.allReduceMin(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                    )
                elif node.op_type == "AllReduceMax":
                    tensors[node.output[0]] = self.handler.allReduceMax(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                    )
                elif node.op_type == "AllReduceAvg":
                    tensors[node.output[0]] = self.handler.allReduceAvg(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                    )
                elif node.op_type == "AllGather":
                    for name, tensor in zip(
                        node.output,
                        self.handler.allGather(
                            tensors[node.input[0]],
                            None,
                            len(node.output),
                        ),
                    ):
                        tensors[name] = tensor
                elif node.op_type == "Broadcast":
                    tensors[node.output[0]] = self.handler.broadcast(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                        next(
                            (attr.i for attr in node.attribute if attr.name == "root"),
                            0,
                        ),
                    )
                elif node.op_type == "Expand":
                    shape = _parse_data(data[node.input[1]])
                    tensors[node.output[0]] = self.handler.expand(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                        shape,
                    )
                elif node.op_type == "Erf":
                    tensors[node.output[0]] = self.handler.erf(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                    )
                elif node.op_type == "Where":
                    tensors[node.output[0]] = self.handler.where(
                        tensors[node.input[1]],
                        tensors[node.input[2]],
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                    )
                else:
                    raise Exception('Unsupported operator "{}"'.format(node.op_type))
                new_node_name.append(node.name)
            # update the node_list
            node_list = list(set(node_name) - set(new_node_name))

        ################################
        # Set weight tensors as persistent
        ################################
        for name, obj in tensors.items():
            if data.get(name) != None:
                obj.set_persistent()

        ################################
        # Allocate memory space for data
        ################################
        self.handler.data_malloc()

        #################################
        # Copy in data to tensor objects
        #################################
        for name, obj in tensors.items():
            tensor = data.get(name)
            if tensor == None:
                if any(input.name == name for input in model.graph.input):
                    self.inputs[name] = obj
            else:
                self.initializer[obj.fuid()] = tensor
                # TODO: delete these lines after copyin_numpy is stable
                # if tensor.data_type == TensorProto.INT32:
                #     obj.copyin_int32(_parse_data(tensor))
                # elif tensor.data_type == TensorProto.INT64:
                #     obj.copyin_int64(_parse_data(tensor))
                # elif tensor.data_type == TensorProto.FLOAT:
                #     obj.copyin_float(_parse_data(tensor))
                # elif tensor.data_type == TensorProto.BOOL:
                #     obj.copyin_int8(_parse_data(tensor))
                # elif tensor.data_type == TensorProto.FLOAT16:
                #     obj.copyin_float16(_parse_data_fp16(tensor))
                # elif tensor.data_type == TensorProto.INT8:
                #     obj.copyin_uint8(_parse_data(tensor))
                # elif tensor.data_type == TensorProto.BFLOAT16:
                #     obj.copyin_float16(_parse_data_fp16(tensor))
                # else:
                #     assert False, "Unsupported Tensor Type: {}".format(tensor.data_type)
                obj.copyin_numpy(to_array(tensor))

        for output in model.graph.output:
            self.outputs[output.name] = tensors[output.name]

    def to_onnx(self, name: str) -> ModelProto:
        class Context:
            # saves object names, including tensors and operators
            names: Dict[Union[backend.Tensor, backend.Operator], str] = dict()
            # counts the occurrence times of each operator for naming
            count_op: Dict[backend.OpTypeId, int] = dict()
            # counts input and output tensors for naming
            count_in, count_out = 0, 0
            # saves nodes (operators)
            nodes: List[NodeProto] = []
            # saves global input tensors
            inputs: List[ValueInfoProto] = []
            # saves global output tensors
            outputs: List[ValueInfoProto] = []
            # saves global input tensors
            initializers: List[TensorProto] = []

            def name_op(self, op: backend.Operator) -> Tuple[backend.OpTypeId, str]:
                ty = op.op_type().id()
                name = "{}{}".format(ty.name, self.count_op.setdefault(ty, 0) + 1)
                self.names[op] = name
                self.count_op[ty] += 1
                return ty, name

            def push_output(self, name: str, tensor: backend.Tensor) -> str:
                self.names[tensor] = name
                if not tensor.has_target():
                    shape = tensor.shape()
                    dtype = backend.tensor_dtype(tensor)
                    value_info = make_tensor_value_info(name, dtype, shape)
                    check_value_info(value_info)
                    self.outputs.append(value_info)
                return name

            def push_input(
                self, tensor: backend.Tensor, init: Optional[TensorProto]
            ) -> str:
                name = self.names.get(tensor)
                # means that this input is a global input
                if name is None:
                    self.count_in += 1
                    name = "input{}".format(self.count_in)
                    self.names[tensor] = name
                    if init != None:
                        init.name = name
                        self.initializers.append(init)
                    else:
                        shape = tensor.shape()
                        dtype = backend.tensor_dtype(tensor)
                        value_info = make_tensor_value_info(name, dtype, shape)
                        check_value_info(value_info)
                        self.inputs.append(value_info)
                return name

            def push_data_input(
                self,
                node_name: str,
                attr_name: str,
                elem_type: int,
                shape: Sequence[int],
                vals: Any,
            ) -> str:
                name = "{}_{}".format(node_name, attr_name)
                tensor = make_tensor(name, elem_type, shape, vals)
                check_tensor(tensor)
                self.initializers.append(tensor)
                return name

            def push_node(self, node: NodeProto) -> None:
                check_node(node)
                self.nodes.append(node)

            def build(self, name: str) -> ModelProto:
                graph = make_graph(
                    self.nodes, name, self.inputs, self.outputs, self.initializers
                )
                check_graph(graph)

                model = make_model(graph)
                check_model(model)

                return model

        # 拓扑排序
        if not self.handler.topo_sort():
            raise Exception("Sorting fails")

        ops = self.handler.operators()  # 图中所有算子（节点）

        ctx = Context()

        for op in ops:
            ty, name = ctx.name_op(op)
            inputs = [
                ctx.push_input(it, self.initializer.get(it.fuid()))
                for it in op.inputs()
            ]
            outputs = [
                ctx.push_output("{}_{}".format(name, i), it)
                for (i, it) in enumerate(op.outputs())
            ]
            if ty == backend.OpTypeId.Conv:
                ph, pw, dh, dw, sh, sw = backend.conv_attrs_of(op)
                ctx.push_node(
                    make_node(
                        ty.name,
                        inputs,
                        outputs,
                        name,
                        pads=[ph, pw, ph, pw],
                        strides=[sh, sw],
                        dilations=[dh, dw],
                        group=op.inputs()[0].shape()[1] // op.inputs()[1].shape()[1],
                    )
                )
            elif ty == backend.OpTypeId.ConvTranspose:
                ph, pw, sh, sw, dh, dw, oph, opw = backend.conv_trans_attrs_of(op)
                ctx.push_node(
                    make_node(
                        ty.name,
                        inputs,
                        outputs,
                        name,
                        pads=[ph, pw],
                        strides=[sh, sw],
                        dilations=[dh, dw],
                        output_padding=[oph, opw],
                    )
                )
            elif ty == backend.OpTypeId.MatMul:
                transA, transB = backend.matmul_attrs_of(op)
                ctx.push_node(
                    make_node(
                        "Gemm", inputs, outputs, name, transA=transA, transB=transB
                    )
                )
            elif ty == backend.OpTypeId.BatchNormalization:
                inputs = [inputs[i] for i in [0, 3, 4, 1, 2]]
                momentum, eps, training = backend.batch_norm_attrs_of(op)
                ctx.push_node(
                    make_node(
                        "BatchNormalization",
                        inputs,
                        outputs,
                        name,
                        epsilon=eps,
                        momentum=momentum,
                        training_mode=training,
                    )
                )
            elif ty == backend.OpTypeId.MaxPool:
                kh, kw, dh, dw, ph, pw, sh, sw = backend.pool_attrs_of(op)
                ctx.push_node(
                    make_node(
                        ty.name,
                        inputs,
                        outputs,
                        name,
                        kernel_shape=[kh, kw],
                        pads=[ph, pw, ph, pw],
                        dilations=[dh, dw],
                        strides=[sh, sw],
                    )
                )
            elif ty == backend.OpTypeId.AveragePool:
                kh, kw, dh, dw, ph, pw, sh, sw = backend.pool_attrs_of(op)
                ctx.push_node(
                    make_node(
                        "AveragePool",
                        inputs,
                        outputs,
                        name,
                        kernel_shape=[kh, kw],
                        pads=[ph, pw, ph, pw],
                        strides=[sh, sw],
                    )
                )
            elif ty in [
                backend.OpTypeId.Add,
                backend.OpTypeId.Sub,
                backend.OpTypeId.Mul,
                backend.OpTypeId.Div,
                backend.OpTypeId.Pow,
                backend.OpTypeId.Relu,
                backend.OpTypeId.Sigmoid,
                backend.OpTypeId.Tanh,
                backend.OpTypeId.Softmax,
                backend.OpTypeId.Abs,
                backend.OpTypeId.Identity,
                backend.OpTypeId.PRelu,
                backend.OpTypeId.Sqrt,
                backend.OpTypeId.Erf,
            ]:
                ctx.push_node(make_node(ty.name, inputs, outputs, name))
            elif ty == backend.OpTypeId.Flatten:
                axis = backend.flatten_axis_of(op)
                ctx.push_node(make_node(ty.name, inputs, outputs, name, axis=axis))
            elif ty == backend.OpTypeId.Transpose:
                perm = backend.transpose_permute_of(op)
                ctx.push_node(make_node(ty.name, inputs, outputs, name, perm=perm))
            elif ty == backend.OpTypeId.Reshape:
                shape = backend.reshape_shape_of(op)
                inputs.append(
                    ctx.push_data_input(
                        name,
                        "shape",
                        TensorProto.INT64,
                        [len(shape)],
                        shape,
                    )
                )
                ctx.push_node(make_node(ty.name, inputs, outputs, name))
            elif ty == backend.OpTypeId.Concat:
                axis = backend.concat_axis_of(op)
                ctx.push_node(make_node(ty.name, inputs, outputs, name, axis=axis))
            elif ty == backend.OpTypeId.Split:
                axis = backend.split_axis_of(op)
                num_outputs = len(outputs)
                split = op.inputs()[0].shape()[axis] // num_outputs
                inputs.append(
                    ctx.push_data_input(
                        name,
                        "split",
                        TensorProto.INT64,
                        [len(outputs)],
                        [split for _ in range(0, num_outputs)],
                    )
                )
                ctx.push_node(
                    make_node(
                        ty.name,
                        inputs,
                        outputs,
                        name,
                        axis=axis,
                    )
                )
            elif ty == backend.OpTypeId.Gather:
                axis = backend.gather_axis_of(op)
                ctx.push_node(make_node(ty.name, inputs, outputs, name, axis=axis))
            elif ty == backend.OpTypeId.ReduceMean:
                axes, keepdims = backend.reduce_mean_attrs_of(op)
                inputs.append(
                    ctx.push_data_input(
                        name, "axes", TensorProto.INT64, [len(axes)], axes
                    )
                )
                ctx.push_node(
                    make_node(ty.name, inputs, outputs, name, keepdims=keepdims)
                )
            elif ty == backend.OpTypeId.Slice:
                raise Exception("TODO")
            elif ty == backend.OpTypeId.Pad:
                pads = backend.pad_pads_of(op)
                inputs.append(
                    ctx.push_data_input(
                        name, "pads", TensorProto.INT64, [len(pads)], pads
                    )
                )
                ctx.push_node(make_node(ty.name, inputs, outputs, name))
            elif ty == backend.OpTypeId.Clip:
                min, max = backend.clip_attrs_of(op)
                if min != None:
                    inputs.append(
                        ctx.push_data_input(name, "min", TensorProto.FLOAT, [], [min])
                    )
                else:
                    inputs.append(
                        ctx.push_data_input(name, "min", TensorProto.FLOAT, [], [])
                    )
                if max != None:
                    inputs.append(
                        ctx.push_data_input(name, "max", TensorProto.FLOAT, [], [max])
                    )
                else:
                    inputs.append(
                        ctx.push_data_input(name, "max", TensorProto.FLOAT, [], [])
                    )
                ctx.push_node(make_node(ty.name, inputs, outputs, name))
            elif ty == backend.OpTypeId.Cast:
                to = backend.cast_to_of(op)
                ctx.push_node(make_node(ty.name, inputs, outputs, name, to=to))
            elif ty == backend.OpTypeId.Where:
                assert len(inputs) == 3, "Check Where Op must have three inputs."
                new_inputs = [inputs[2], inputs[0], inputs[1]]
                ctx.push_node(make_node(ty.name, new_inputs, outputs, name))
            elif ty == backend.OpTypeId.Expand:
                shape = backend.expand_shape_of(op)
                ctx.push_node(make_node(ty.name, inputs, outputs, name, shape=shape))
            else:
                raise Exception("Unsupported OpType", ty)

        return ctx.build(name)

    def init(self) -> None:
        self.handler.data_malloc()

    def optimize(self) -> None:
        self.handler.optimize()

    def set_input(self, inputShapes: List[int]) -> None:
        for newInput, oldInput in zip(inputShapes, self.inputs):
            oldTensor = self.inputs[oldInput]
            self.handler.change_shape(newInput, oldTensor.fuid())
        self.handler.shape_infer()
        self.handler.data_malloc()

    def getShape(self, name: str) -> List[int]:
        if name in self.inputs:
            ans = self.handler.getDims(self.inputs[name])
        else:
            ans = self.handler.getDims(self.outputs[name])
        return ans

    def tune(self) -> None:
        self.handler.tune()

    def run(self) -> None:
        self.handler.run()

    def get_perf_time(self) -> float:
        self.handler.get_perf_time()


def from_onnx(model: ModelProto, runtime):
    stub = OnnxStub(model, runtime)
    return stub.inputs, stub.outputs, stub.handler


def _search_shape(model: ModelProto, name: str) -> List[int]:
    ans = (
        next(
            (
                [
                    (d.dim_value if d.dim_value > 0 else 1)
                    for d in tensor.type.tensor_type.shape.dim
                ]
                for tensor in model.graph.value_info
                if tensor.name == name
            ),
            None,
        )
        or next(
            (
                [
                    (d.dim_value if d.dim_value > 0 else 1)
                    for d in tensor.type.tensor_type.shape.dim
                ]
                for tensor in model.graph.input
                if tensor.name == name
            ),
            None,
        )
        or next(
            (
                [
                    (d.dim_value if d.dim_value > 0 else 1)
                    for d in tensor.type.tensor_type.shape.dim
                ]
                for tensor in model.graph.output
                if tensor.name == name
            ),
            None,
        )
        or next(
            [int(d) for d in tensor.dims]
            for tensor in model.graph.initializer
            if tensor.name == name
        )
    )
    return ans


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


def _parse_data(tensor: TensorProto) -> List[Any]:
    return to_array(tensor).flatten().tolist()


def _parse_data_fp16(tensor: TensorProto):
    list_ = []
    if len(tensor.int32_data) != 0:
        for element_data in tensor.int32_data:
            element_byte = element_data.to_bytes(2, "little")
            list_.append(element_byte[0] + element_byte[1] * 256)
    elif len(tensor.raw_data) != 0:
        list_raw_data = list(tensor.raw_data)
        list_data = [list_raw_data[i : i + 2] for i in range(0, len(list_raw_data), 2)]
        for ele in list_data:
            list_.append(ele[0] + ele[1] * 256)
    else:
        raise Exception("Tensor have no float16 data!")
    return list_


def _take_shape_dim(shape: TensorShapeProto) -> List[int]:
    return [(d.dim_value if d.dim_value > 0 else 1) for d in shape.dim]


def _analyse_node(node: NodeProto, tensors) -> bool:
    for i in node.input:
        if i not in tensors:
            return True
    return False
