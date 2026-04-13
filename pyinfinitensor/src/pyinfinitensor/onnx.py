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
    ValidationError,
)
from onnx.shape_inference import infer_shapes
from onnx.numpy_helper import to_array, from_array
from typing import Dict, List, Any, Tuple, Sequence, Union, Optional
from functools import reduce
from onnxsim import simplify
import copy
import warnings
import numpy as np


class OnnxStub:
    """
    The Onnx model imported into infinitensor.
    It can be generated from an Onnx model object.
    """

    def __init__(
        self,
        model: ModelProto,
        runtime,
        use_naive_allocator: bool = False,
        matmul_compute_type: str = "default",
    ):
        # We use some user-defined operators for distributed inference
        try:
            # onnx simplifier performs inplace simplify
            model_simp, check = simplify(copy.deepcopy(model))
            if check:
                model = model_simp
        except ValidationError:
            pass
        except RuntimeError:
            pass

        self.inputs: Dict[str, backend.Tensor] = {}
        self.outputs: Dict[str, backend.Tensor] = {}
        self.tensors: Dict[str, backend.Tensor] = {}
        self.tensor_node_map: Dict[str, str] = {}
        self.initializer: Dict[int, TensorProto] = {}
        self.use_naive_allocator: bool = use_naive_allocator
        # try:
        #     model = infer_shapes(model)
        # except:
        #     warnings.warn("infer_shapes failed.")
        self.handler = backend.GraphHandler(runtime)

        # 处理重名和匿名算子
        names = {}
        for node in model.graph.node:
            if node.name == "":
                node.name = "missing_name(" + node.op_type + ")"
            if node.name in names:
                names[node.name] += 1
                node.name += "_" + str(names[node.name])
            else:
                names[node.name] = 0
        # 拓扑排序
        sorted_nodes = []
        known_edge = set(t.name for t in model.graph.input)
        known_edge.update(t.name for t in model.graph.initializer)
        while len(sorted_nodes) < len(model.graph.node):
            updated = False
            for i, node in enumerate(model.graph.node):
                # TODO：only consider the case where the input of resize exist emptyInput
                if all(t in known_edge or t == "" for t in node.input):
                    node.name = str(len(sorted_nodes)) + "_" + node.name
                    sorted_nodes.append(i)
                    known_edge.update(node.output)
                    for t_ in node.output:
                        self.tensor_node_map[t_] = node.name
                    updated = True
            if not updated:
                raise Exception("Graph has cycle")

        tensors: Dict[str, backend.Tensor] = dict()
        data: Dict[str, TensorProto] = dict()

        for initializer in model.graph.initializer:
            dims = [d for d in initializer.dims]
            tensors[initializer.name] = self.handler.tensor(dims, initializer.data_type)
            data[initializer.name] = initializer
            tensors[initializer.name].set_weight()

        # print("Available data keys:", list(data.keys()))
        # print("Node inputs:", node.input)
        # print("Looking for key:", node.input[1])

        for input in model.graph.input:
            dims = _take_shape_dim(input.type.tensor_type.shape)
            if input.name not in tensors.keys():
                tensors[input.name] = self.handler.tensor(
                    dims, input.type.tensor_type.elem_type
                )
                tensors[input.name].set_input()

        for node_idx in sorted_nodes:
            node = model.graph.node[node_idx]
            try:
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
                                    tensors[node.input[2]].shape(),
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
                elif node.op_type == "Elu":
                    attributes = _parse_attribute(node, {"alpha": 1.0})
                    alpha = attributes["alpha"]
                    tensors[node.output[0]] = self.handler.elu(
                        tensors[node.input[0]], tensors.get(node.output[0]), alpha
                    )
                elif node.op_type == "ConvTranspose":
                    attributes = _parse_attribute(
                        node,
                        {
                            "dilations": [1, 1],
                            "pads": [0, 0, 0, 0],
                            "strides": [1, 1],
                            "output_padding": [0, 0],
                        },
                    )
                    (d, p, s, op) = (
                        attributes[name]
                        for name in ["dilations", "pads", "strides", "output_padding"]
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
                        tensors[bias] = self.handler.convTransposed2d(
                            tensors[adapt],
                            tensors[node.input[1]],
                            None,
                            p[0],
                            p[1],
                            s[0],
                            s[1],
                            d[0],
                            d[1],
                            op[0],
                            op[1],
                        )
                        tensors[reshape] = self.handler.reshape(
                            tensors[node.input[2]],
                            None,
                            [
                                1,
                                reduce(
                                    lambda acc, x: acc * x,
                                    tensors[node.input[2]].shape(),
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
                        tensors[node.output[0]] = self.handler.convTransposed2d(
                            tensors[adapt],
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
                        tensors[node.input[0]],  # input
                        tensors[node.input[1]],  # weight
                        tensors.get(node.output[0]),
                        False,
                        False,
                        None,
                        backend.ActType.Linear,
                        matmul_compute_type,
                    )
                elif node.op_type == "Gemm":
                    attributes = _parse_attribute(
                        node, {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0}
                    )
                    (alpha, beta, transA, transB) = (
                        attributes[name] for name in ["alpha", "beta", "transA", "transB"]
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
                        matmul_compute_type,
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
                elif node.op_type == "LayerNormalization":
                    (input, scale) = (tensors[node.input[i]] for i in [0, 1])
                    bias = None if len(node.input) < 3 else tensors[node.input[2]]
                    output = tensors.get(node.output[0])
                    attributes = _parse_attribute(
                        node, {"axis": -1, "epsilon": 1e-05, "stash_type": 1}
                    )
                    (axis, eps, stash_type) = (
                        attributes[name] for name in ["axis", "epsilon", "stash_type"]
                    )
                    tensors[node.output[0]] = self.handler.layerNormalization(
                        input,
                        scale,
                        output,
                        bias,
                        eps,
                        axis,
                        stash_type,
                    )
                elif node.op_type == "InstanceNormalization":
                    (input, scale, bias) = (tensors[node.input[i]] for i in [0, 1, 2])

                    output = tensors.get(node.output[0])

                    tensors[node.output[0]] = self.handler.instanceNormalization(
                        input,
                        output,
                        scale,
                        bias,
                        next(
                            (attr.f for attr in node.attribute if attr.name == "epsilon"),
                            1e-5,
                        ),
                    )
                elif node.op_type == "RMSNorm":
                    tensors[node.output[0]] = self.handler.RMSNorm(
                        tensors[node.input[0]],
                        tensors[node.input[1]],
                        tensors.get(node.output[0]),
                    )
                elif node.op_type == "MaxPool":
                    attributes = _parse_attribute(
                        node,
                        {
                            "kernel_shape": None,
                            "dilations": [1, 1],
                            "pads": [0, 0, 0, 0],
                            "strides": [1, 1],
                            "ceil_mode": 0,
                        },
                    )
                    (k, d, p, s, ceil_mode) = (
                        attributes[name]
                        for name in [
                            "kernel_shape",
                            "dilations",
                            "pads",
                            "strides",
                            "ceil_mode",
                        ]
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
                            ceil_mode,
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
                            ceil_mode,
                        )
                elif node.op_type == "AveragePool":
                    attributes = _parse_attribute(
                        node,
                        {
                            "kernel_shape": None,
                            "pads": [0, 0, 0, 0],
                            "strides": [1, 1],
                            "ceil_mode": 0,
                        },
                    )
                    (k, p, s, ceil_mode) = (
                        attributes[name]
                        for name in ["kernel_shape", "pads", "strides", "ceil_mode"]
                    )

                    # Avg Pool 1D
                    if len(p) == 2:
                        tensors[node.output[0]] = self.handler.avgPool(
                            tensors[node.input[0]],
                            tensors.get(node.output[0]),
                            1,
                            k[0],
                            1,
                            1,
                            0,
                            p[0],
                            1,
                            s[0],
                            ceil_mode,
                        )
                    elif p[0] != p[2] or p[1] != p[3]:
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
                            ceil_mode,
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
                            ceil_mode,
                        )
                elif node.op_type == "GlobalAveragePool":
                    [_, _, h, w] = tensors[node.input[0]].shape()
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
                        0,
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
                elif node.op_type == "LeakyRelu":
                    tensors[node.output[0]] = self.handler.leakyRelu(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                        next(
                            (attr.f for attr in node.attribute if attr.name == "alpha"),
                            0.01,
                        ),
                    )
                elif node.op_type == "Silu":
                    tensors[node.output[0]] = self.handler.silu(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                    )
                elif node.op_type == "Gelu":
                    tensors[node.output[0]] = self.handler.gelu(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                    )
                elif node.op_type == "Sigmoid":
                    tensors[node.output[0]] = self.handler.sigmoid(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                    )
                elif node.op_type == "HardSigmoid":
                    tensors[node.output[0]] = self.handler.hardSigmoid(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                    )
                elif node.op_type == "HardSwish":
                    tensors[node.output[0]] = self.handler.hardSwish(
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
                elif node.op_type == "Neg":
                    tensors[node.output[0]] = self.handler.neg(
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
                        (
                            next(_parse_data(data[node.input[1]]).__iter__(), None)
                            if len(node.input) > 1
                            else None
                        ),
                        (
                            next(_parse_data(data[node.input[2]]).__iter__(), None)
                            if len(node.input) > 2
                            else None
                        ),
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
                elif node.op_type == "DepthToSpace":
                    blocksize = next(
                        (attr.i for attr in node.attribute if attr.name == "blocksize"),
                        None,
                    )
                    mode = next(
                        (attr.s for attr in node.attribute if attr.name == "mode"),
                        None,
                    )
                    tensors[node.output[0]] = self.handler.depthToSpace(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                        blocksize,
                        mode,
                    )
                elif node.op_type == "Reshape":
                    shape = _parse_data(data[node.input[1]])
                    tensors[node.output[0]] = self.handler.reshape(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                        shape,
                    )
                elif node.op_type == "Resize":
                    output = tensors.get(node.output[0])
                    attributes = _parse_attribute(
                        node,
                        {
                            "antialias": 0,
                            "axes": None,
                            "coordinate_transformation_mode": "half_pixel",
                            "cubic_coeff_a": -0.75,
                            "exclude_outside": 0,
                            "extrapolation_value": 0.0,
                            "keep_aspect_ratio_policy": "stretch",
                            "mode": "nearest",
                            "nearest_mode": "none",
                        },
                    )
                    (
                        axes,
                        keep_aspect_ratio_policy,
                        coordinate_transformation_mode,
                        mode,
                        nearest_mode,
                    ) = (
                        attributes[name]
                        for name in [
                            "axes",
                            "keep_aspect_ratio_policy",
                            "coordinate_transformation_mode",
                            "mode",
                            "nearest_mode",
                        ]
                    )
                    if len(node.input) > 1 and node.input[1] in data:
                        roiVal = _parse_data(data[node.input[1]])
                    else:
                        roiVal = []
                    if len(node.input) > 2 and node.input[2] in data:
                        scalesVal = _parse_data(data[node.input[2]])
                    else:
                        scalesVal = []
                    if len(node.input) > 3 and node.input[3] in data:
                        sizesVal = _parse_data(data[node.input[3]])
                    else:
                        sizesVal = []
                    tensors[node.output[0]] = self.handler.resize(
                        tensors[node.input[0]],
                        output,
                        axes,
                        (
                            tensors[node.input[3]]
                            if len(node.input) > 3 and node.input[3] != ""
                            else None
                        ),
                        (
                            tensors[node.input[2]]
                            if len(node.input) > 2 and node.input[2] != ""
                            else None
                        ),
                        (
                            tensors[node.input[1]]
                            if len(node.input) > 1 and node.input[1] != ""
                            else None
                        ),
                        sizesVal,
                        scalesVal,
                        roiVal,
                        mode,
                        keep_aspect_ratio_policy,
                        nearest_mode,
                        coordinate_transformation_mode,
                    )
                elif node.op_type == "Squeeze":
                    axes = _parse_data(data[node.input[1]]) if len(node.input) > 1 else None
                    if axes is None:
                        axes = next(
                            (attr.ints for attr in node.attribute if attr.name == "axes"),
                            [],
                        )
                    tensors[node.output[0]] = self.handler.squeeze(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                        axes,
                    )
                elif node.op_type == "Unsqueeze":
                    axes = _parse_data(data[node.input[1]]) if len(node.input) > 1 else None
                    if axes is None:
                        axes = next(
                            (attr.ints for attr in node.attribute if attr.name == "axes")
                        )
                    tensors[node.output[0]] = self.handler.unsqueeze(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                        axes,
                    )
                elif node.op_type == "Concat":
                    tensors[node.output[0]] = self.handler.concat(
                        [tensors[name] for name in node.input],
                        tensors.get(node.output[0]),
                        next((attr.i for attr in node.attribute if attr.name == "axis")),
                    )
                elif node.op_type == "AttentionKVCache":
                    tensors[node.output[0]] = self.handler.attentionKVCache(
                        tensors[node.input[0]],
                        tensors[node.input[1]],
                        tensors[node.input[2]],
                        tensors[node.input[3]],
                        tensors[node.input[4]],
                        tensors[node.input[5]],
                        tensors.get(node.output[0]),
                    )
                elif node.op_type == "RoPE":
                    tensors[node.output[0]] = self.handler.RoPE(
                        tensors[node.input[0]],
                        tensors[node.input[1]],
                        tensors.get(node.output[0]),
                    )
                elif node.op_type == "Split":
                    split = (
                        _parse_data(data[node.input[1]]) if (len(node.input) > 1) else None
                    )
                    if split is None:
                        split = next(
                            (attr.ints for attr in node.attribute if attr.name == "split"),
                            None,
                        )
                    for name, tensor in zip(
                        node.output,
                        self.handler.split(
                            tensors[node.input[0]],
                            None,
                            next(
                                (attr.i for attr in node.attribute if attr.name == "axis"),
                                0,
                            ),
                            split if split is not None else len(node.output),
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
                elif node.op_type == "GatherElements":
                    tensors[node.output[0]] = self.handler.gatherElements(
                        tensors[node.input[0]],
                        tensors[node.input[1]],
                        tensors.get(node.output[0]),
                        next(
                            (attr.i for attr in node.attribute if attr.name == "axis"),
                            0,
                        ),
                    )
                elif node.op_type == "ReduceMean":
                    tensors[node.output[0]] = self.handler.reduceMean(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                        # NOTE(constroy): `axes` is an attribute until opset version 13.
                        next(
                            (attr.ints for attr in node.attribute if attr.name == "axes"),
                            None,
                        ),
                        next(
                            (attr.i for attr in node.attribute if attr.name == "keepdims"),
                            1,
                        )
                        != 0,
                    )
                elif node.op_type == "Slice":

                    def clamp(nums):
                        MAX_INT = 0x7FFFFFFF
                        return [min(x, MAX_INT) for x in nums]

                    tensors[node.output[0]] = self.handler.slice(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                        clamp(_parse_data(data[node.input[1]])),
                        clamp(_parse_data(data[node.input[2]])),
                        (
                            clamp(_parse_data(data[node.input[3]]))
                            if len(node.input) > 3
                            else None
                        ),
                        (
                            clamp(_parse_data(data[node.input[4]]))
                            if len(node.input) > 4
                            else None
                        ),
                    )
                elif node.op_type == "Pad":
                    tensors[node.output[0]] = self.handler.pad(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                        _parse_data(data[node.input[1]]),
                        _parse_data(data[node.input[3]]) if len(node.input) > 3 else None,
                    )
                elif node.op_type == "Dropout":
                    for name, tensor in zip(
                        node.output,
                        self.handler.dropout(
                            tensors[node.input[0]],
                            tensors.get(node.output[0]),
                            tensors.get(node.output[1]) if len(node.output) > 1 else None,
                            (
                                _parse_data(data[node.input[1]])[0]
                                if len(node.input) > 1
                                else 0.5
                            ),
                            (
                                _parse_data(data[node.input[2]])[0]
                                if len(node.input) > 2
                                else False
                            ),
                        ),
                    ):
                        tensors[name] = tensor
                elif node.op_type == "Cast":
                    tensors[node.output[0]] = self.handler.cast(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                        next((attr.i for attr in node.attribute if attr.name == "to")),
                    )
                elif node.op_type == "ReduceSum":
                    if any(attr.name == "communicator" for attr in node.attribute):
                        # ReduceSum with communicator is treated as allReduceSum.
                        tensors[node.output[0]] = self.handler.allReduceSum(
                            tensors[node.input[0]],
                            tensors.get(node.output[0]),
                        )
                    else:
                        # NOTE: `axes` is an attribute until opset version 13.
                        if len(node.input) > 1:
                            axis = _parse_data(data[node.input[1]])
                        else:
                            axis = next(
                                (
                                    attr.ints
                                    for attr in node.attribute
                                    if attr.name == "axes"
                                ),
                                None,
                            )
                        keepdims = (
                            next(
                                (
                                    attr.i
                                    for attr in node.attribute
                                    if attr.name == "keepdims"
                                ),
                                1,
                            )
                            != 0
                        )

                        tensors[node.output[0]] = self.handler.reduceSum(
                            tensors[node.input[0]],
                            tensors.get(node.output[0]),
                            axis,
                            keepdims,
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
                elif node.op_type == "Send":
                    source = next(
                        (attr.i for attr in node.attribute if attr.name == "source"),
                        0,
                    )
                    destination = next(
                        (attr.i for attr in node.attribute if attr.name == "destination"),
                        0,
                    )

                    self.handler.send(
                        tensors[node.input[0]],
                        source,
                        destination,
                        None,
                    )
                elif node.op_type == "Recv":
                    source = next(
                        (attr.i for attr in node.attribute if attr.name == "source"),
                        0,
                    )
                    destination = next(
                        (attr.i for attr in node.attribute if attr.name == "destination"),
                        0,
                    )

                    for attr in node.attribute:
                        if attr.name == "shape":
                            shapeBasic = attr.ints
                    shape = []
                    for item in shapeBasic:
                        shape.append(item)

                    for attr in node.attribute:
                        if attr.name == "dataType":
                            outputType = attr.i
                    tensors[node.output[0]] = self.handler.recv(
                        tensors.get(node.output[0]),
                        source,
                        destination,
                        shape,
                        outputType,
                        None,
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
                    ## If Y is single -inf, treat Where as Add
                    ## TODO: deal with cases where Y is single inf or 0
                    if node.input[0] in data and node.input[2] in data:
                        where_condition = to_array(data[node.input[0]])
                        where_alt = to_array(data[node.input[2]])
                        if where_alt.size == 1:
                            if np.isneginf(where_alt) or np.all(where_alt < -3e38):
                                node.input[0] = node.input[0] + "_alt"
                                if node.input[0] not in data:
                                    where_value = np.where(
                                        where_condition, 0, -np.inf
                                    ).astype(where_alt.dtype)
                                    data[node.input[0]] = from_array(
                                        where_value, node.input[0]
                                    )
                                    tensors[node.input[0]] = self.handler.tensor(
                                        list(where_value.shape),
                                        data[node.input[0]].data_type,
                                    )
                                    tensors[node.input[0]].set_weight()
                                tensors[node.output[0]] = self.handler.add(
                                    tensors[node.input[1]],
                                    tensors[node.input[0]],
                                    tensors.get(node.output[0]),
                                )
                                continue
                    tensors[node.output[0]] = self.handler.where(
                        tensors[node.input[1]],
                        tensors[node.input[2]],
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                    )
                elif node.op_type in ["Constant", "ConstantOfShape"]:
                    output_name = node.output[0]
                    attributes = _parse_attribute(node)
                    tensor = attributes["value"]
                    dims = [d for d in tensor.dims]
                    tensors[output_name] = self.handler.tensor(dims, tensor.data_type)
                    data[output_name] = tensor
                    tensors[output_name].set_weight()
                elif node.op_type == "LRN":
                    attributes = _parse_attribute(
                        node, {"alpha": 0.0001, "beta": 0.75, "bias": 1.0, "size": 1}
                    )
                    (alpha, beta, bias, size) = (
                        attributes[name] for name in ["alpha", "beta", "bias", "size"]
                    )
                    tensors[node.output[0]] = self.handler.lrn(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                        alpha,
                        beta,
                        bias,
                        size,
                    )
                elif node.op_type == "ArgMax":
                    attributes = _parse_attribute(
                        node, {"axis": -1, "keepdims": 1, "select_last_index": 0}
                    )
                    if attributes["keepdims"] == 0:
                        attributes["keepdims"] = False
                    else:
                        attributes["keepdims"] = True
                    tensors[node.output[0]] = self.handler.argmax(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                        attributes["axis"],
                        attributes["keepdims"],       
                    )
                else:
                    raise Exception('Unsupported operator "{}"'.format(node.op_type))
            except Exception as e:
                print('Error occurred while processing node "{}": {}'.format(node.name, str(e)))
        for output in model.graph.output:
            tensors[output.name].set_output()
        ################################
        # Allocate memory space for data
        ################################
        self.handler.data_malloc(self.use_naive_allocator)

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

        for name, obj in tensors.items():
            self.tensors[name] = obj

        for output in model.graph.output:
            print(f"Output: {output.name}")
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
            elif ty == backend.OpTypeId.Elu:
                alpha = backend.elu_alpha_of(op)
                ctx.push_node(make_node("Elu", inputs, outputs, name, alpha=alpha))
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
                kh, kw, dh, dw, ph, pw, sh, sw, ceil_mode = backend.pool_attrs_of(op)
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
                        ceil_mode=ceil_mode,
                    )
                )
            elif ty == backend.OpTypeId.AveragePool:
                kh, kw, dh, dw, ph, pw, sh, sw, ceil_mode = backend.pool_attrs_of(op)
                ctx.push_node(
                    make_node(
                        "AveragePool",
                        inputs,
                        outputs,
                        name,
                        kernel_shape=[kh, kw],
                        pads=[ph, pw, ph, pw],
                        strides=[sh, sw],
                        ceil_mode=ceil_mode,
                    )
                )
            elif ty in [
                backend.OpTypeId.Add,
                backend.OpTypeId.Sub,
                backend.OpTypeId.Mul,
                backend.OpTypeId.Div,
                backend.OpTypeId.Pow,
                backend.OpTypeId.Relu,
                backend.OpTypeId.Gelu,
                backend.OpTypeId.Sigmoid,
                backend.OpTypeId.HardSigmoid,
                backend.OpTypeId.HardSwish,
                backend.OpTypeId.Tanh,
                backend.OpTypeId.Softmax,
                backend.OpTypeId.Abs,
                backend.OpTypeId.Identity,
                backend.OpTypeId.PRelu,
                backend.OpTypeId.Sqrt,
                backend.OpTypeId.Erf,
                backend.OpTypeId.Neg,
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
            elif ty == backend.OpTypeId.Squeeze:
                axes = backend.squeeze_axes_of(op)
                inputs.append(
                    ctx.push_data_input(
                        name,
                        "axes",
                        TensorProto.INT64,
                        [len(axes)],
                        axes,
                    )
                )
                ctx.push_node(make_node(ty.name, inputs, outputs, name))
            elif ty == backend.OpTypeId.Unsqueeze:
                axes = backend.unsqueeze_axes_of(op)
                inputs.append(
                    ctx.push_data_input(
                        name,
                        "axes",
                        TensorProto.INT64,
                        [len(axes)],
                        axes,
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
            elif ty in [backend.OpTypeId.ReduceMean, backend.OpTypeId.ReduceSum]:
                axes, keepdims = backend.reduce_attrs_of(op)
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
            elif ty == backend.OpTypeId.LRN:
                alpha, beta, bias, size = backend.lrn_attrs_of(op)
                ctx.push_node(
                    make_node(
                        ty.name,
                        inputs,
                        outputs,
                        name,
                        alpha,
                        beta,
                        bias,
                        size,
                    )
                )
            else:
                raise Exception("Unsupported OpType", ty)

        return ctx.build(name)

    def init(self) -> None:
        self.handler.data_malloc(self.use_naive_allocator)

    def optimize(self) -> None:
        self.handler.optimize()

    def clone_KV(self, tensor: backend.Tensor) -> backend.Tensor:
        return self.handler.clone_KV(tensor)

    def free_heap(self) -> None:
        self.handler.free_heap()

    def set_input(self, inputShapes: List[int]) -> None:
        for newInput, oldInput in zip(inputShapes, self.inputs):
            oldTensor = self.inputs[oldInput]
            self.handler.change_shape(newInput, oldTensor.fuid())
        self.handler.shape_infer()
        self.handler.data_malloc(self.use_naive_allocator)

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

    def run_with_cudagraph(self) -> None:
        self.handler.run_with_cudagraph()

    def get_perf_time(self) -> float:
        self.handler.get_perf_time()


def from_onnx(model: ModelProto, runtime):
    stub = OnnxStub(model, runtime)
    return stub.inputs, stub.outputs, stub.handler


def _parse_attribute(node: NodeProto, attrs: Dict[str, Any] = dict()) -> Dict[str, Any]:
    for attr in node.attribute:
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
