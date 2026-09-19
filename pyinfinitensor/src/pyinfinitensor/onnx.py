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
        input_shapes: Optional[Dict[str, Sequence[int]]] = None,
    ):
        model = copy.deepcopy(model)
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
        # Which handing-out of weight storage the initializers were last written
        # into. `None` until they have been written at all.
        self._weights_written_at: Optional[int] = None
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
            # A small integer constant may take part in a shape computation, so
            # hand its contents over before any operator is built: shape
            # inference runs while the graph is being constructed, and the
            # initializer data itself is only copied in much later, by `init`.
            _seed_shape_value(tensors[initializer.name], initializer)

        graph_input_names = {input.name for input in model.graph.input}
        input_shapes = {} if input_shapes is None else dict(input_shapes)
        unknown_shapes = set(input_shapes).difference(graph_input_names)
        if unknown_shapes:
            raise ValueError(
                "input_shapes names are not graph inputs: {}".format(
                    ", ".join(sorted(unknown_shapes))
                )
            )

        for input in model.graph.input:
            shape_proto = input.type.tensor_type.shape
            dims = _take_shape_dim(shape_proto)
            if input.name not in tensors.keys():
                tensors[input.name] = self.handler.tensor(
                    dims, input.type.tensor_type.elem_type
                )
                tensors[input.name].set_input()
                # Keep which dimensions the model declared dynamic, so that
                # later shape changes can reject illegal ones.
                descs = _take_dim_descs(shape_proto)
                if descs:
                    self.handler.set_dim_descs(descs, tensors[input.name].fuid())

        # Operators infer their output shapes while they are constructed. A
        # symbolic ONNX dimension has only the placeholder one at that point,
        # which may be too small for a valid geometry chain. Let callers seed
        # a legal concrete realization before the first operator is made; later
        # set_input calls still use the same DimDesc validation and may vary
        # every dynamic dimension.
        for name, shape in input_shapes.items():
            try:
                self.handler.change_shape(list(shape), tensors[name].fuid())
            except RuntimeError as e:
                raise RuntimeError('input_shapes["{}"]: {}'.format(name, e)) from e

        for node_idx in sorted_nodes:
            node = model.graph.node[node_idx]
            if node.op_type == "Conv":
                attributes = _parse_attribute(
                    node,
                    {
                        "dilations": [1, 1],
                        "pads": [0, 0, 0, 0],
                        "strides": [1, 1],
                    },
                )
                d, p, s = (
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
                d, p, s, op = (
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
                alpha, beta, transA, transB = (
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
                input, mean, var, scale, bias = (
                    tensors[node.input[i]] for i in [0, 3, 4, 1, 2]
                )
                output = tensors.get(node.output[0])
                attributes = _parse_attribute(
                    node, {"momentum": 0.9, "epsilon": 1e-05, "training_mode": 0}
                )
                momentum, eps, training = (
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
                input, scale = (tensors[node.input[i]] for i in [0, 1])
                bias = None if len(node.input) < 3 else tensors[node.input[2]]
                output = tensors.get(node.output[0])
                attributes = _parse_attribute(
                    node, {"axis": -1, "epsilon": 1e-05, "stash_type": 1}
                )
                axis, eps, stash_type = (
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
                input, scale, bias = (tensors[node.input[i]] for i in [0, 1, 2])

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
                k, d, p, s, ceil_mode = (
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
                k, p, s, ceil_mode = (
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
                # The window is the whole input by definition, so it is not
                # passed: a spatial size read here would be the placeholder a
                # dynamic dimension carries before any real shape arrives, and a
                # window of that pools one element and returns the input as it
                # was. `globalWindow` has the operator read the size off each
                # input instead.
                input_shape = tensors[node.input[0]].shape()
                if len(input_shape) == 3:
                    h, w = 1, input_shape[-1]
                elif len(input_shape) == 4:
                    h, w = input_shape[-2:]
                else:
                    raise ValueError(
                        "GlobalAveragePool input must have rank 3 or 4, got {}".format(
                            len(input_shape)
                        )
                    )
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
                    True,
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
                if _has_input(node, 1) and node.input[1] not in data:
                    # The target shape is worked out from the shape of an
                    # input, so wire it up as an edge of the graph instead of
                    # reading a constant that is not there.
                    tensors[node.output[0]] = self.handler.reshape_with_shape_input(
                        tensors[node.input[0]],
                        tensors[node.input[1]],
                        tensors.get(node.output[0]),
                    )
                else:
                    shape = _parse_static_input(data, node, 1, required=True)
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
                    MIN_INT = -0x80000000
                    return [max(MIN_INT, min(x, MAX_INT)) for x in nums]

                # Which axes are sliced, and at what step, is written as a
                # constant even by an exporter that computes the bounds: the
                # bounds move with the shape while the axes do not.
                axes = (
                    clamp(_parse_static_input(data, node, 3))
                    if _has_input(node, 3)
                    else None
                )
                steps = (
                    clamp(_parse_static_input(data, node, 4))
                    if _has_input(node, 4)
                    else None
                )
                computed = (node.input[1] not in data) or (node.input[2] not in data)
                if computed:
                    # A bound is worked out from the shape of an input, so wire
                    # both up as edges of the graph instead of reading constants
                    # that are not there. A constant bound is an edge too, and
                    # carries its value as a shape value.
                    tensors[node.output[0]] = self.handler.slice_with_bound_inputs(
                        tensors[node.input[0]],
                        tensors[node.input[1]],
                        tensors[node.input[2]],
                        tensors.get(node.output[0]),
                        axes,
                        steps,
                    )
                else:
                    tensors[node.output[0]] = self.handler.slice(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                        clamp(_parse_static_input(data, node, 1, required=True)),
                        clamp(_parse_static_input(data, node, 2, required=True)),
                        axes,
                        steps,
                    )
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
                    raise NotImplementedError("Dropout training mode is not supported")
                if len(node.output) > 1 and node.output[1]:
                    raise NotImplementedError("Dropout mask output is not supported")
                tensors[node.output[0]] = self.handler.identity(
                    tensors[node.input[0]], tensors.get(node.output[0])
                )
            elif node.op_type == "Cast":
                to = next(attr.i for attr in node.attribute if attr.name == "to")
                if to == tensors[node.input[0]].dtype():
                    # A cast to the type a tensor already holds does nothing.
                    # An exporter emits these freely, and around a shape
                    # subgraph in particular, so reading one as a copy keeps
                    # the chain of shape values unbroken. Same idiom as
                    # `Dropout` above, which is also a copy at inference.
                    tensors[node.output[0]] = self.handler.identity(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                    )
                else:
                    tensors[node.output[0]] = self.handler.cast(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                        to,
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
                if _has_input(node, 1) and node.input[1] not in data:
                    # The target is worked out from the shape of an input, so
                    # wire it up as an edge of the graph instead of reading a
                    # constant that is not there.
                    tensors[node.output[0]] = self.handler.expand_with_shape_input(
                        tensors[node.input[0]],
                        tensors[node.input[1]],
                        tensors.get(node.output[0]),
                    )
                else:
                    shape = _parse_static_input(data, node, 1, required=True)
                    tensors[node.output[0]] = self.handler.expand(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                        shape,
                    )
            elif node.op_type == "Tile":
                if node.input[1] not in data:
                    # The counts are worked out from the shape of an input, so
                    # wire them up as an edge of the graph instead of reading a
                    # constant that is not there.
                    tensors[node.output[0]] = self.handler.tile_with_repeats_input(
                        tensors[node.input[0]],
                        tensors[node.input[1]],
                        tensors.get(node.output[0]),
                    )
                else:
                    repeats = _parse_static_input(data, node, 1, required=True)
                    tensors[node.output[0]] = self.handler.tile(
                        tensors[node.input[0]],
                        tensors.get(node.output[0]),
                        repeats,
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
            elif node.op_type == "Constant":
                output_name = node.output[0]
                attributes = _parse_attribute(node)
                tensor = attributes["value"]
                dims = [d for d in tensor.dims]
                tensors[output_name] = self.handler.tensor(dims, tensor.data_type)
                data[output_name] = tensor
                tensors[output_name].set_weight()
                # An axis or a shape reaches a graph either as an initializer or
                # as a `Constant`, and an exporter may choose either. Seed both
                # the same way; see the loop over `model.graph.initializer`.
                _seed_shape_value(tensors[output_name], tensor)
            elif node.op_type == "ConstantOfShape":
                raise NotImplementedError(
                    'Unsupported operator "ConstantOfShape": static and dynamic '
                    "shape execution is not implemented"
                )
            elif node.op_type == "LRN":
                attributes = _parse_attribute(
                    node, {"alpha": 0.0001, "beta": 0.75, "bias": 1.0, "size": 1}
                )
                alpha, beta, bias, size = (
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
                raise Exception('Unsupported operator "{}"'.format(node.op_type))

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

    def to_onnx(self, name: str) -> ModelProto:
        class Context:
            def __init__(self):
                self.names: Dict[Union[backend.Tensor, backend.Operator], str] = {}
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
                (
                    kh,
                    kw,
                    dh,
                    dw,
                    ph,
                    pw,
                    sh,
                    sw,
                    ceil_mode,
                    _,
                ) = backend.pool_attrs_of(op)
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
                (
                    kh,
                    kw,
                    dh,
                    dw,
                    ph,
                    pw,
                    sh,
                    sw,
                    ceil_mode,
                    global_window,
                ) = backend.pool_attrs_of(op)
                if global_window:
                    # The window is the whole input, which is what
                    # `GlobalAveragePool` means and what no window size can say.
                    # Writing the size held right now would fix the pool to the
                    # last shape it was given.
                    ctx.push_node(make_node("GlobalAveragePool", inputs, outputs, name))
                else:
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
                backend.OpTypeId.Shape,
            ]:
                ctx.push_node(make_node(ty.name, inputs, outputs, name))
            elif ty == backend.OpTypeId.Softmax:
                axis = backend.softmax_axis_of(op)
                ctx.push_node(make_node(ty.name, inputs, outputs, name, axis=axis))
            elif ty == backend.OpTypeId.Flatten:
                axis = backend.flatten_axis_of(op)
                ctx.push_node(make_node(ty.name, inputs, outputs, name, axis=axis))
            elif ty == backend.OpTypeId.Transpose:
                perm = backend.transpose_permute_of(op)
                ctx.push_node(make_node(ty.name, inputs, outputs, name, perm=perm))
            elif ty == backend.OpTypeId.Reshape:
                # A Reshape reading its target from an edge already has the
                # second input ONNX asks for; only the other shape has to spell
                # the target out as a constant.
                if len(inputs) == 1:
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
                        ctx.push_data_input(name, "min", input_dtype, [], [min_value])
                    )
                elif max_value is not None:
                    inputs.append("")
                if max_value is not None:
                    inputs.append(
                        ctx.push_data_input(name, "max", input_dtype, [], [max_value])
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
                # As with Reshape: one reading its target from an edge already
                # has the second input ONNX asks for.
                if len(inputs) == 1:
                    shape = backend.expand_shape_of(op)
                    inputs.append(
                        ctx.push_data_input(
                            name, "shape", TensorProto.INT64, [len(shape)], shape
                        )
                    )
                ctx.push_node(make_node(ty.name, inputs, outputs, name))
            elif ty == backend.OpTypeId.Tile:
                # ONNX always asks for the counts as a second input, so one that
                # holds them as a constant has to write them out.
                if len(inputs) == 1:
                    repeats = backend.tile_repeats_of(op)
                    inputs.append(
                        ctx.push_data_input(
                            name,
                            "repeats",
                            TensorProto.INT64,
                            [len(repeats)],
                            repeats,
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

    def init(self) -> None:
        self.handler.data_malloc(self.use_naive_allocator)
        self._copy_initializers()

    def _copy_initializers(self) -> None:
        """Write the weights the model came with into the graph.

        Weight storage is handed out once and then left alone however often the
        shape changes afterwards, so what was written into it stands and writing
        it again produces what is already there. Reading a weight out of the
        model is not cheap -- protobuf to numpy, then a copy into the graph --
        and on a model of any size it is the greater part of what a shape change
        costs, so the work is skipped rather than its result cached.

        The graph says when it hands weight storage out afresh, which is the one
        thing that invalidates what was written: a failed allocation rolls back
        to none of it being held, and the weights want writing again after. A
        shape change on its own moves activations only and does not count.
        """
        generation = self.handler.weight_data_generation()
        if self._weights_written_at == generation:
            return
        for name, initializer in self._initializer_by_name.items():
            self.tensors[name].copyin_numpy(to_array(initializer))
        self._weights_written_at = generation

    def optimize(self) -> None:
        self.handler.optimize()
        self._forget_folded_tensors()

    def shape_subgraph_size(self) -> int:
        """How many operators describe shapes rather than data.

        This is the part of the graph `fold_shape_subgraph` can reach, so
        reading it before a fold says what the fold started from.
        """
        return self.handler.shape_subgraph_size()

    def fold_shape_subgraph(self) -> int:
        """Replace the settled part of the shape subgraph with constants.

        A dimension the model declared fixed cannot change, so whatever follows
        only from such dimensions is already its final value and need not be
        worked out again on every inference. Returns how many operators this
        removed. Dimensions the model declared dynamic are left alone, along
        with everything reading them.
        """
        dropped = self.handler.fold_fixed_shape_subgraph()
        self._forget_folded_tensors()
        return dropped

    def merge_duplicate_shape_operators(self) -> int:
        """Leave one of each shape computation the graph works out twice over.

        An exporter that reads a dimension off the same tensor in several places
        emits the read several times, and each copy is then run on every
        inference. Two operators of the same kind, reading the very same tensors
        with the same attributes, produce the same answer under every shape, so
        whoever read the copy is pointed at the one that stays. Returns how many
        operators this removed.

        Tensors are compared as objects rather than by what they currently hold,
        so two that happen to carry equal numbers under this shape are not taken
        for one another. This is separate from `fold_shape_subgraph`: that one
        removes a computation whose answer has settled, and this one removes a
        second copy of a computation that has not.
        """
        merged = self.handler.merge_duplicate_shape_operators()
        self._forget_folded_tensors()
        return merged

    def pin_dims(self, name: str, axes: Sequence[int]) -> None:
        """Declare that some of an input's dimensions are fixed from here on.

        A model is commonly exported with a dimension left dynamic and then
        deployed at one size for good: an export makes the batch variable, and
        the server that loads it serves one batch size. Saying so lets
        everything following from that dimension be worked out once by
        `fold_shape_subgraph`, which is knowledge no simplifier run before the
        deployment existed could have had.

        The pinned dimensions keep the size they hold now, and asking for
        another one afterwards is refused.
        """
        if name not in self.inputs:
            raise ValueError(
                'no input named "{}"; this model takes {}'.format(
                    name, ", ".join(self.inputs) or "nothing"
                )
            )
        tensor = self.inputs[name]
        rank = len(tensor.shape())
        descs = list(tensor.dim_descs())
        if not descs:
            # A tensor that declared nothing counts every dimension dynamic.
            descs = [backend.DimDesc(True, "") for _ in range(rank)]
        for axis in axes:
            if not 0 <= axis < rank:
                raise ValueError(
                    'input "{}" has rank {}, so it has no axis {}'.format(
                        name, rank, axis
                    )
                )
            descs[axis] = backend.DimDesc(False, "")
        self.handler.set_dim_descs(descs, tensor.fuid())
        # Spreading this fixedness through the shape values takes a round of
        # inference, and `set_input` will not do it: it skips a shape the
        # tensor already carries, which is exactly the size just pinned.
        self.handler.shape_infer()

    def _forget_folded_tensors(self) -> None:
        """Drop what the graph no longer holds.

        A constant that took part only in a settled shape computation is gone
        from the graph along with the operators that read it. Copying its
        contents in again would write into memory the graph has stopped
        accounting for, so it is taken off the list of things to copy.
        """
        gone = set(self.handler.folded_away_tensors())
        if not gone:
            return
        for name in [n for n, t in self.tensors.items() if t.fuid() in gone]:
            self._initializer_by_name.pop(name, None)
        for fuid in gone:
            self.initializer.pop(fuid, None)

    def clone_KV(self, tensor: backend.Tensor) -> backend.Tensor:
        return self.handler.clone_KV(tensor)

    def free_heap(self) -> None:
        self.handler.free_heap()

    def trim_memory(self) -> None:
        self.handler.trim_memory()

    def activation_allocations(self) -> int:
        """How many times activation storage has been asked of the runtime.

        Storage that is reused does not count, so this is what a workload
        changing shape from one inference to the next is trying to hold down.
        """
        return self.handler.activation_allocations()

    def activation_peak(self) -> int:
        """Bytes the activations need for the shape currently in place."""
        return self.handler.activation_peak()

    def activation_capacity(self) -> int:
        """Bytes actually held for activations.

        This stands at the high watermark of the shapes seen so far; the
        difference from `activation_peak` is the slack that buys the reuse.
        """
        return self.handler.activation_capacity()

    def allocated_bytes(self) -> int:
        """Every byte the graph holds: activations, weights and heap."""
        return self.handler.allocated_bytes()

    def set_input(self, inputShapes: List[Sequence[int]]) -> None:
        if len(inputShapes) != len(self.inputs):
            raise ValueError(
                "inputShapes must contain one shape per model input; expected "
                "{}, got {}".format(len(self.inputs), len(inputShapes))
            )
        # Shapes that are already in place need nothing done to them, and one
        # shape is commonly asked for many times over -- a batch size that
        # holds from one inference to the next. Laying the memory out again
        # costs more than running the whole graph, so it is worth not doing.
        # A shape equal to the one a tensor already carries is valid by having
        # been accepted once, so nothing is skipped but the work itself.
        if all(
            list(requested) == self.inputs[name].shape()
            for requested, name in zip(inputShapes, self.inputs)
        ):
            return
        changed = []
        for newInput, oldInput in zip(inputShapes, self.inputs):
            oldTensor = self.inputs[oldInput]
            previous = oldTensor.shape()
            try:
                self.handler.change_shape(newInput, oldTensor.fuid())
            except (RuntimeError, TypeError) as e:
                # change_shape validates before mutation. No inference or
                # allocation has happened yet, so restore earlier inputs to
                # keep their metadata consistent with the existing storage.
                for tensor, shape in reversed(changed):
                    self.handler.change_shape(shape, tensor.fuid())
                # Name the offending input; the backend only knows tensor ids.
                if isinstance(e, TypeError):
                    raise
                raise RuntimeError('input "{}": {}'.format(oldInput, e)) from e
            changed.append((oldTensor, previous))
        self.handler.shape_infer()
        self.init()

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
                "{} input {} is required and must be constant".format(
                    node.op_type, index
                )
            )
        return None

    name = node.input[index]
    if name not in data:
        raise ValueError(
            '{} input {} ("{}") must be constant'.format(node.op_type, index, name)
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
            "{} input {} must contain exactly one value".format(node.op_type, index)
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


# A tensor taking part in a shape computation holds one entry per dimension, so
# it is tiny. The bound keeps large integer tensors, which are data rather than
# shapes, from being copied for nothing.
_SHAPE_VALUE_MAX_ELEMENTS = 64
_SHAPE_VALUE_DTYPES = (TensorProto.INT32, TensorProto.INT64)


def _seed_shape_value(tensor: backend.Tensor, initializer: TensorProto) -> None:
    """Give the backend the contents of a small integer constant."""
    if initializer.data_type not in _SHAPE_VALUE_DTYPES:
        return
    if len(initializer.dims) > 1:
        return
    values = to_array(initializer).reshape(-1)
    if values.size > _SHAPE_VALUE_MAX_ELEMENTS:
        return
    tensor.set_shape_value([int(v) for v in values])


def _take_dim_descs(shape: TensorShapeProto) -> List[backend.DimDesc]:
    """Describe each dimension of `shape` as fixed, symbolic or unknown.

    Returns an empty list when every dimension is fixed, which tells the
    backend that the tensor never declared its dynamicity and that its shape
    may still be replaced as a whole.
    """
    descs = []
    for d in shape.dim:
        if d.dim_value > 0:
            descs.append(backend.DimDesc(False))
        else:
            # `dim_param` names a symbolic dimension; without it the dimension
            # is dynamic but anonymous.
            descs.append(backend.DimDesc(True, d.dim_param))
    return descs if any(desc.dynamic for desc in descs) else []
