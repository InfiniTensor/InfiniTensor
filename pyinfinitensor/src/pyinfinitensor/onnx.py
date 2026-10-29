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
try:
    import onnxruntime as _onnxruntime
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "pyinfinitensor ONNX import requires onnxruntime; install the "
        "declared pyinfinitensor dependencies before importing OnnxStub"
    ) from error
from onnxsim import simplify
import copy
import operator
import warnings
import numpy as np
import time
import json


from .shape import ShapeValueEvaluator, _onnx_numpy_dtype, _shape_control_node_indexes


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
        model = copy.deepcopy(model)
        dynamic_template = any(
            _shape_is_dynamic(_shape_spec(item.type.tensor_type.shape))
            for item in model.graph.input
        )
        # Preserve symbolic input dimensions and avoid simplifier verification
        # forwards during dynamic-model loading. Static import stays compatible.
        if not dynamic_template:
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
        self._initializer_by_name: Dict[str, TensorProto] = {}
        self.use_naive_allocator: bool = use_naive_allocator
        self._input_shape_specs = {
            item.name: _shape_spec(item.type.tensor_type.shape)
            for item in model.graph.input
            if item.name not in {initializer.name for initializer in model.graph.initializer}
        }
        self._shape_evaluator = ShapeValueEvaluator(model)
        self._shape_control_nodes = _shape_control_node_indexes(model)
        self._dynamic_bindings: List[Dict[str, Any]] = []
        self._prepared_input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None
        self._prepared_is_valid = False
        self._weights_uploaded = False
        self._weight_copy_count = 0
        self.last_timing: Dict[str, float] = {}
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
        pending_nodes = set(range(len(model.graph.node)))
        known_edge = set(t.name for t in model.graph.input)
        known_edge.update(t.name for t in model.graph.initializer)
        while pending_nodes:
            updated = False
            for i, node in enumerate(model.graph.node):
                if i not in pending_nodes:
                    continue
                # TODO：only consider the case where the input of resize exist emptyInput
                if all(t in known_edge or t == "" for t in node.input):
                    node.name = str(len(sorted_nodes)) + "_" + node.name
                    sorted_nodes.append(i)
                    pending_nodes.remove(i)
                    known_edge.update(node.output)
                    for t_ in node.output:
                        self.tensor_node_map[t_] = node.name
                    updated = True
            if not updated:
                unresolved = []
                for i in sorted(pending_nodes):
                    node = model.graph.node[i]
                    missing = [
                        name for name in node.input if name and name not in known_edge
                    ]
                    unresolved.append(
                        "{} ({}) missing {}".format(
                            node.name or node.op_type, node.op_type, missing
                        )
                    )
                raise ValueError(
                    "Unable to resolve ONNX graph; it contains a cycle or "
                    "missing inputs: {}".format("; ".join(unresolved))
                )

        tensors: Dict[str, backend.Tensor] = dict()
        data: Dict[str, TensorProto] = dict()

        for initializer in model.graph.initializer:
            dims = [d for d in initializer.dims]
            tensors[initializer.name] = self.handler.tensor(dims, initializer.data_type)
            data[initializer.name] = initializer
            tensors[initializer.name].set_weight()

        for input in model.graph.input:
            dims = _take_shape_dim(input.type.tensor_type.shape)
            if input.name not in tensors.keys():
                tensors[input.name] = self.handler.tensor(
                    dims, input.type.tensor_type.elem_type
                )
                tensors[input.name].set_input()

        # The first import uses a concrete placeholder only to construct the
        # graph.  Its symbolic dimensions remain in _input_shape_specs and are
        # rebound by prepare() before each actual request.
        self._shape_evaluator.evaluate(
            {name: _take_shape_dim_from_spec(spec) for name, spec in self._input_shape_specs.items()}
        )

        for node_idx in sorted_nodes:
            node = model.graph.node[node_idx]
            if node_idx in self._shape_control_nodes and node.op_type != "Constant":
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
                alpha = next(
                    (attr.f for attr in node.attribute if attr.name == "alpha"),
                    0.2,
                )
                beta = next(
                    (attr.f for attr in node.attribute if attr.name == "beta"),
                    0.5,
                )
                tensors[node.output[0]] = self.handler.hardSigmoid(
                    tensors[node.input[0]],
                    tensors.get(node.output[0]),
                    alpha, beta,
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
                    _parse_static_scalar(data, node, 1),
                    _parse_static_scalar(data, node, 2),
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
                shape, dynamic = self._parse_shape_parameter(data, node, 1, required=True)
                tensors[node.output[0]] = self.handler.reshape(
                    tensors[node.input[0]],
                    tensors.get(node.output[0]),
                    shape,
                )
                if dynamic:
                    self._dynamic_bindings.append({
                        "kind": "reshape", "output": tensors[node.output[0]].fuid(),
                        "params": [node.input[1]], "node": node.name,
                    })
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
                axes = _parse_static_input(data, node, 1)
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
                axes = _parse_static_input(data, node, 1)
                if axes is None:
                    axes = next(
                        (attr.ints for attr in node.attribute if attr.name == "axes"),
                        None,
                    )
                if axes is None:
                    raise ValueError("Unsqueeze requires constant axes")
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
                split = _parse_static_input(data, node, 1)
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
                # ONNX opset 13+: `axes` is a second input, not an attribute.
                # PyTorch 2.x exports all ReduceMean nodes with the new spec.
                axes = next(
                    (attr.ints for attr in node.attribute if attr.name == "axes"),
                    None,
                )
                if axes is None and len(node.input) > 1 and node.input[1]:
                    axes = _parse_static_input(data, node, 1)
                tensors[node.output[0]] = self.handler.reduceMean(
                    tensors[node.input[0]],
                    tensors.get(node.output[0]),
                    axes,
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

                starts, starts_dynamic = self._parse_shape_parameter(data, node, 1, required=True)
                ends, ends_dynamic = self._parse_shape_parameter(data, node, 2, required=True)
                axes, axes_dynamic = self._parse_shape_parameter(data, node, 3)
                steps, steps_dynamic = self._parse_shape_parameter(data, node, 4)
                tensors[node.output[0]] = self.handler.slice(
                    tensors[node.input[0]],
                    tensors.get(node.output[0]),
                    clamp(starts), clamp(ends), clamp(axes) if axes is not None else None,
                    clamp(steps) if steps is not None else None,
                )
                if starts_dynamic or ends_dynamic or axes_dynamic or steps_dynamic:
                    self._dynamic_bindings.append({
                        "kind": "slice", "output": tensors[node.output[0]].fuid(),
                        "params": [node.input[i] if _has_input(node, i) else "" for i in range(1, 5)], "node": node.name,
                    })
            elif node.op_type == "Pad":
                attributes = _parse_attribute(node, {"mode": b"constant"})
                mode = attributes["mode"]
                if mode not in ["constant", b"constant"]:
                    raise NotImplementedError(
                        'Pad mode "{}" is not supported'.format(
                            mode.decode() if isinstance(mode, bytes) else mode
                        )
                    )
                constant_value = _parse_static_scalar(data, node, 2)
                if constant_value not in [None, 0, 0.0, False]:
                    raise NotImplementedError(
                        "Pad only supports a constant value of zero"
                    )
                tensors[node.output[0]] = self.handler.pad(
                    tensors[node.input[0]],
                    tensors.get(node.output[0]),
                    _parse_static_input(data, node, 1, required=True),
                    _parse_static_input(data, node, 3),
                )
            elif node.op_type == "Dropout":
                training_mode = _parse_static_scalar(data, node, 2)
                if training_mode:
                    raise NotImplementedError(
                        "Dropout training mode is not supported"
                    )
                if len(node.output) > 1 and node.output[1]:
                    raise NotImplementedError(
                        "Dropout mask output is not supported"
                    )
                tensors[node.output[0]] = self.handler.identity(
                    tensors[node.input[0]], tensors.get(node.output[0])
                )
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
                    if _has_input(node, 1):
                        axis = _parse_static_input(data, node, 1)
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
                shape, dynamic = self._parse_shape_parameter(data, node, 1, required=True)
                tensors[node.output[0]] = self.handler.expand(
                    tensors[node.input[0]],
                    tensors.get(node.output[0]),
                    shape,
                )
                if dynamic:
                    self._dynamic_bindings.append({
                        "kind": "expand", "output": tensors[node.output[0]].fuid(),
                        "params": [node.input[1]], "node": node.name,
                    })
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
            elif node.op_type == "Constant":
                output_name = node.output[0]
                attributes = _parse_attribute(node)
                tensor = attributes["value"]
                dims = [d for d in tensor.dims]
                tensors[output_name] = self.handler.tensor(dims, tensor.data_type)
                data[output_name] = tensor
                tensors[output_name].set_weight()
            elif node.op_type == "ConstantOfShape":
                raise NotImplementedError(
                    'Unsupported operator "ConstantOfShape": static and dynamic '
                    "shape execution is not implemented"
                )
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
            else:
                raise NotImplementedError(
                    'Unsupported ONNX operator "{}" at node "{}"; inputs={}, outputs={}'.format(
                        node.op_type, node.name or node.op_type,
                        list(node.input), list(node.output),
                    )
                )

        for output in model.graph.output:
            tensors[output.name].set_output()
        for name, obj in tensors.items():
            tensor = data.get(name)
            if tensor is None:
                if any(input.name == name for input in model.graph.input):
                    self.inputs[name] = obj
            else:
                self.initializer[obj.fuid()] = tensor
                self._initializer_by_name[name] = tensor

        for name, obj in tensors.items():
            self.tensors[name] = obj

        for output in model.graph.output:
            self.outputs[output.name] = tensors[output.name]

        self.init()
        if not any(_shape_is_dynamic(spec) for spec in self._input_shape_specs.values()):
            self._prepared_input_shapes = {
                name: tuple(_take_shape_dim_from_spec(spec))
                for name, spec in self._input_shape_specs.items()
            }
            self._prepared_is_valid = True

    def to_onnx(self, name: str) -> ModelProto:
        class Context:
            def __init__(self):
                self.names: Dict[
                    Union[backend.Tensor, backend.Operator], str
                ] = {}
                self.count_op: Dict[backend.OpTypeId, int] = {}
                self.count_in = 0
                self.count_out = 0
                self.nodes: List[NodeProto] = []
                self.inputs: List[ValueInfoProto] = []
                self.outputs: List[ValueInfoProto] = []
                self.initializers: List[TensorProto] = []

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
                    if init is not None:
                        init = copy.deepcopy(init)
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
                ph, pw, dh, dw, sh, sw, oph, opw = backend.conv_trans_attrs_of(op)
                ctx.push_node(
                    make_node(
                        ty.name,
                        inputs,
                        outputs,
                        name,
                        pads=[ph, pw, ph, pw],
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
                backend.OpTypeId.Abs,
                backend.OpTypeId.Identity,
                backend.OpTypeId.PRelu,
                backend.OpTypeId.Sqrt,
                backend.OpTypeId.Erf,
                backend.OpTypeId.Neg,
            ]:
                ctx.push_node(make_node(ty.name, inputs, outputs, name))
            elif ty == backend.OpTypeId.Softmax:
                axis = backend.softmax_axis_of(op)
                ctx.push_node(
                    make_node(ty.name, inputs, outputs, name, axis=axis)
                )
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
                split = [output.shape()[axis] for output in op.outputs()]
                inputs.append(
                    ctx.push_data_input(
                        name,
                        "split",
                        TensorProto.INT64,
                        [len(split)],
                        split,
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
                min_value, max_value = backend.clip_attrs_of(op)
                input_dtype = backend.tensor_dtype(op.inputs()[0])
                if min_value is not None:
                    inputs.append(
                        ctx.push_data_input(
                            name, "min", input_dtype, [], [min_value]
                        )
                    )
                elif max_value is not None:
                    inputs.append("")
                if max_value is not None:
                    inputs.append(
                        ctx.push_data_input(
                            name, "max", input_dtype, [], [max_value]
                        )
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
                inputs.append(
                    ctx.push_data_input(
                        name, "shape", TensorProto.INT64, [len(shape)], shape
                    )
                )
                ctx.push_node(make_node(ty.name, inputs, outputs, name))
            elif ty == backend.OpTypeId.LRN:
                alpha, beta, bias, size = backend.lrn_attrs_of(op)
                ctx.push_node(
                    make_node(
                        ty.name,
                        inputs,
                        outputs,
                        name,
                        alpha=alpha,
                        beta=beta,
                        bias=bias,
                        size=size,
                    )
                )
            else:
                raise Exception("Unsupported OpType", ty)

        return ctx.build(name)

    def init(self, reload_weights: bool = False) -> None:
        """Allocate graph storage, uploading initializers only once by default."""
        self.handler.data_malloc(self.use_naive_allocator)
        if reload_weights or not self._weights_uploaded:
            self._copy_initializers()
            self._weights_uploaded = True

    def _copy_initializers(self) -> None:
        for name, initializer in self._initializer_by_name.items():
            self.tensors[name].copyin_numpy(to_array(initializer))
            self._weight_upload_bytes = (
                getattr(self, "_weight_upload_bytes", 0) + to_array(initializer).nbytes
            )
        self._weight_copy_count += 1

    @property
    def weight_upload_count(self) -> int:
        """Number of initializer upload passes, useful for lifecycle tests."""
        return self._weight_copy_count

    @property
    def weight_upload_bytes(self) -> int:
        return getattr(self, "_weight_upload_bytes", 0)

    @property
    def weight_addresses(self) -> Dict[str, int]:
        return {name: self.tensors[name].data_ptr() for name in self._initializer_by_name}

    def optimize(self) -> None:
        self.handler.optimize()

    def clone_KV(self, tensor: backend.Tensor) -> backend.Tensor:
        return self.handler.clone_KV(tensor)

    def free_heap(self) -> None:
        self.handler.free_heap()

    def trim_memory(self) -> None:
        self.handler.trim_memory()


    def _parse_shape_parameter(
        self,
        data: Dict[str, TensorProto],
        node: NodeProto,
        index: int,
        required: bool = False,
    ) -> Tuple[Optional[List[int]], bool]:
        if not _has_input(node, index):
            if required:
                raise ValueError("{} input {} is required".format(node.op_type, index))
            return None, False
        name = node.input[index]
        if name in data:
            return [int(item) for item in _parse_data(data[name])], False
        try:
            return self._shape_evaluator.value(name), True
        except ValueError as error:
            raise ValueError(
                '{} node "{}" has unsupported dynamic shape input {} ("{}"): {}'.format(
                    node.op_type, node.name or node.op_type, index, name, error
                )
            ) from error

    def _normalise_input_shapes(
        self, input_shapes: Union[Dict[str, Sequence[int]], Sequence[Sequence[int]]]
    ) -> Dict[str, Tuple[int, ...]]:
        names = list(self.inputs)
        if isinstance(input_shapes, dict):
            if set(input_shapes) != set(names):
                raise ValueError("input_shapes must contain exactly {}".format(names))
            pairs = [(name, input_shapes[name]) for name in names]
        else:
            if len(input_shapes) != len(names):
                raise ValueError(
                    "inputShapes must contain one shape per model input; expected {}, got {}".format(
                        len(names), len(input_shapes)
                    )
                )
            pairs = list(zip(names, input_shapes))
        normalised = {}
        symbols = {}

        def dimension(value):
            if isinstance(value, (bool, np.bool_)):
                raise TypeError("boolean dimension")
            return operator.index(value)

        for name, shape in pairs:
            try:
                dims = tuple(dimension(item) for item in shape)
            except (TypeError, ValueError) as error:
                raise ValueError("invalid shape for {}".format(name)) from error
            spec = self._input_shape_specs.get(name, [])
            if len(dims) != len(spec) or any(item <= 0 or item > np.iinfo(np.int32).max for item in dims):
                raise ValueError(
                    "invalid shape for {}: {}; expected positive int32 rank-{} dimensions".format(
                        name, dims, len(spec)
                    )
                )
            for symbol, dim in zip(spec, dims):
                if isinstance(symbol, str) and symbol:
                    if symbol in symbols and symbols[symbol] != dim:
                        raise ValueError(
                            "inconsistent symbolic dimension {} for input {}".format(
                                symbol, name
                            )
                        )
                    symbols[symbol] = dim
            normalised[name] = dims
        return normalised

    def _update_dynamic_parameters(
        self, shapes: Dict[str, Tuple[int, ...]], *, values_ready=False
    ) -> None:
        if not self._dynamic_bindings:
            return
        if not values_ready:
            self._shape_evaluator.evaluate(shapes)
        for binding in self._dynamic_bindings:
            values = [
                self._shape_evaluator.value(name) if name else None
                for name in binding["params"]
            ]
            try:
                if binding["kind"] == "reshape":
                    self.handler.update_reshape(values[0], binding["output"])
                elif binding["kind"] == "expand":
                    self.handler.update_expand(values[0], binding["output"])
                elif binding["kind"] == "slice":
                    for index in (0, 1):
                        values[index] = [max(-0x80000000, min(0x7FFFFFFF, value)) for value in values[index]]
                    self.handler.update_slice(
                        values[0], values[1], values[2], values[3], binding["output"]
                    )
                else:
                    raise AssertionError("unknown dynamic binding")
            except AttributeError as error:
                raise RuntimeError(
                    "this pyinfinitensor extension lacks dynamic shape support; rebuild InfiniTensor"
                ) from error

    def prepare(
        self, input_shapes: Union[Dict[str, Sequence[int]], Sequence[Sequence[int]]]
    ) -> None:
        """Prepare one resident graph for concrete input shapes without reloading weights."""
        shapes = self._normalise_input_shapes(input_shapes)
        if shapes == self._prepared_input_shapes and self._prepared_is_valid:
            return
        # Shape inference can fail, so do every validation/evaluation before
        # touching tensors.  A failure therefore leaves the old prepared graph usable.
        self._shape_evaluator.evaluate(shapes)
        old_shapes = self._prepared_input_shapes
        old_valid = self._prepared_is_valid
        try:
            for name, dims in shapes.items():
                self.handler.change_shape(list(dims), self.inputs[name].fuid())
            self._update_dynamic_parameters(shapes, values_ready=True)
            self.handler.shape_infer()
            self.handler.data_malloc(self.use_naive_allocator)
            self._prepared_input_shapes = shapes
            self._prepared_is_valid = True
        except Exception:
            self._prepared_is_valid = False
            if old_shapes is not None and old_valid:
                try:
                    for name, dims in old_shapes.items():
                        self.handler.change_shape(list(dims), self.inputs[name].fuid())
                    self._update_dynamic_parameters(old_shapes)
                    self.handler.shape_infer()
                    self.handler.data_malloc(self.use_naive_allocator)
                    self._prepared_input_shapes = old_shapes
                    self._prepared_is_valid = True
                except Exception:
                    self._prepared_input_shapes = None
            raise


    def check_capabilities(self, input_shapes=None):
        """Prepare actual provider instances, without invoking the encoder."""
        if input_shapes is not None:
            self.prepare(input_shapes)
        if not self._prepared_is_valid:
            raise RuntimeError("provide concrete input_shapes before capability inspection")
        report = json.loads(self.handler.check_capabilities())
        names = {tensor.fuid(): name for name, tensor in self.tensors.items()}
        for node in report["nodes"]:
            outputs = [names.get(tensor["id"]) for tensor in node["outputs"]]
            node["onnx_node"] = next((self.tensor_node_map[name] for name in outputs
                                      if name in self.tensor_node_map), None)
        return report

    def set_input(self, inputShapes: List[Sequence[int]]) -> None:
        """Compatibility wrapper for the legacy positional input-shape API."""
        self.prepare(inputShapes)

    def run_numpy(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Validate, prepare, execute and return independently-owned NumPy arrays."""
        if set(inputs) != set(self.inputs):
            raise ValueError("inputs must contain exactly {}".format(list(self.inputs)))
        # np.ascontiguousarray promotes rank-0 arrays to rank 1. Preserve ONNX
        # scalar inputs (for example a scalar Gather index).
        arrays = {name: np.asarray(value) for name, value in inputs.items()}
        arrays = {
            name: value if value.flags.c_contiguous else np.ascontiguousarray(value)
            for name, value in arrays.items()
        }
        for name, array in arrays.items():
            expected = _onnx_numpy_dtype(self.inputs[name].dtype())
            if array.dtype != expected:
                raise TypeError(
                    "input {} must have dtype {}, got {}".format(
                        name, expected, array.dtype
                    )
                )
        shapes = {name: array.shape for name, array in arrays.items()}
        started = time.perf_counter()
        self.prepare(shapes)
        prepared = time.perf_counter()
        for name, array in arrays.items():
            self.inputs[name].copyin_numpy(array)
        copied_in = time.perf_counter()
        self.run()
        ran = time.perf_counter()
        outputs = {
            name: np.array(tensor.copyout_numpy(), copy=True)
            for name, tensor in self.outputs.items()
        }
        copied_out = time.perf_counter()
        self.last_timing = {
            "prepare_s": prepared - started,
            "h2d_s": copied_in - prepared,
            "run_s": ran - copied_in,
            "d2h_s": copied_out - ran,
            "total_s": copied_out - started,
        }
        return outputs

    def getShape(self, name: str) -> List[int]:
        if name in self.inputs:
            ans = self.handler.getDims(self.inputs[name])
        else:
            ans = self.handler.getDims(self.outputs[name])
        return ans

    def tune(self) -> None:
        self.handler.tune()

    def run(self) -> None:
        if not self._prepared_is_valid:
            raise RuntimeError(
                "graph has not been prepared; call prepare() or run_numpy() first"
            )
        self.handler.run()

    def run_with_graph(self) -> None:
        if not self._prepared_is_valid:
            raise RuntimeError("graph has not been prepared; call prepare() first")
        self.handler.run_with_graph()

    def get_perf_time(self) -> float:
        return self.handler.get_perf_time()


def from_onnx(model: ModelProto, runtime):
    stub = OnnxStub(model, runtime)
    return stub.inputs, stub.outputs, stub.handler


def _has_input(node: NodeProto, index: int) -> bool:
    return index < len(node.input) and bool(node.input[index])


def _parse_static_input(
    data: Dict[str, TensorProto],
    node: NodeProto,
    index: int,
    required: bool = False,
) -> Optional[List[Any]]:
    if not _has_input(node, index):
        if required:
            raise ValueError(
                '{} input {} is required and must be constant'.format(
                    node.op_type, index
                )
            )
        return None

    name = node.input[index]
    if name not in data:
        raise ValueError(
            '{} input {} ("{}") must be constant'.format(
                node.op_type, index, name
            )
        )
    return _parse_data(data[name])


def _parse_static_scalar(
    data: Dict[str, TensorProto], node: NodeProto, index: int
) -> Optional[Any]:
    values = _parse_static_input(data, node, index)
    if values is None:
        return None
    if len(values) != 1:
        raise ValueError(
            '{} input {} must contain exactly one value'.format(
                node.op_type, index
            )
        )
    return values[0]


def _parse_attribute(
    node: NodeProto, attrs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    attrs = {} if attrs is None else dict(attrs)
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


def _shape_spec(shape: TensorShapeProto) -> List[Union[int, str, None]]:
    result: List[Union[int, str, None]] = []
    for dim in shape.dim:
        if dim.dim_value > 0:
            result.append(int(dim.dim_value))
        elif dim.dim_param:
            result.append(dim.dim_param)
        else:
            result.append(None)
    return result


def _shape_is_dynamic(spec: Sequence[Union[int, str, None]]) -> bool:
    return any(not isinstance(dim, int) or dim <= 0 for dim in spec)


def _take_shape_dim_from_spec(spec: Sequence[Union[int, str, None]]) -> List[int]:
    return [dim if isinstance(dim, int) and dim > 0 else 1 for dim in spec]
