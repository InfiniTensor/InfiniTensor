import backend
from onnx import (
    ModelProto,
    TensorProto,
    NodeProto,
    AttributeProto,
    TensorShapeProto,
    ValueInfoProto,
    load as load_onnx,
)
from onnx.external_data_helper import uses_external_data
from onnx.helper import (
    make_node,
    make_tensor_value_info,
    make_tensor,
    make_graph,
    make_model,
    tensor_dtype_to_np_dtype,
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
from numbers import Integral
from pathlib import Path
from math import prod
try:
    import onnxruntime as _onnxruntime
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "pyinfinitensor ONNX import requires onnxruntime; install the "
        "declared pyinfinitensor dependencies before importing OnnxStub"
    ) from error
from onnxsim import simplify
import copy
import os
import warnings
import numpy as np

from .external_weights import read_external_array


class OnnxWeightStore:
    """Share read-only initializers across ONNX graphs on one runtime.

    External weight files must remain unchanged for the store's lifetime.
    """

    def __init__(self, runtime):
        self.runtime = runtime
        self._weights = {}

    def _source(self, initializer, directory):
        key = (initializer.name, directory if uses_external_data(initializer) else None)
        record = self._weights.get(key)
        if record is not None and initializer == record[0]:
            return record[1]
        return None

    def _remember(self, initializer, directory, tensor):
        key = (initializer.name, directory if uses_external_data(initializer) else None)
        if key in self._weights:
            return False
        tensor.share_weight_data(tensor)
        self._weights[key] = (initializer, tensor)
        return True


class OnnxStub:
    """
    The Onnx model imported into infinitensor.
    Accepts a ModelProto or the path to an ONNX file, including external weights.
    """

    def __init__(
        self,
        model: Union[ModelProto, str, os.PathLike],
        runtime,
        use_naive_allocator: bool = False,
        matmul_compute_type: str = "default",
        *,
        simplify_model: Optional[bool] = None,
        input_shapes: Optional[Dict[str, Sequence[int]]] = None,
        fold_static_shapes: bool = False,
        external_data_dir: Optional[Union[str, os.PathLike]] = None,
        weight_store: Optional[OnnxWeightStore] = None,
        conv_layout: str = "default",
        fuse_fp16_conv_bias: bool = False,
        fuse_fp32_rmsnorm: bool = False,
        fuse_fp32_gelu: bool = False,
        elide_internal_copies: bool = False,
    ):
        if conv_layout not in {"default", "channels_last", "channels_last_silu"}:
            raise ValueError(f"Unknown Conv layout: {conv_layout}")
        if fuse_fp16_conv_bias and conv_layout == "default":
            raise ValueError("FP16 Conv bias fusion requires channels-last Conv")
        if weight_store is not None and weight_store.runtime is not runtime:
            raise ValueError("The ONNX weight store must use the same runtime")
        self._external_data_dir = None
        if isinstance(model, (str, os.PathLike)):
            path = Path(model).resolve()
            model = load_onnx(path, load_external_data=False)
            if external_data_dir is None:
                external_data_dir = path.parent
        else:
            model = copy.deepcopy(model)
        if any(uses_external_data(value) for value in model.graph.initializer):
            if external_data_dir is None:
                raise ValueError("External ONNX weights require a model path or external_data_dir")
            self._external_data_dir = str(Path(external_data_dir).resolve())
            if simplify_model:
                raise ValueError("ONNX simplification is not supported for lazy external weights")
            simplify_model = False
            # Small constants may participate in importer shape evaluation.
            for value in model.graph.initializer:
                if uses_external_data(value) and prod(value.dims) <= 65536:
                    array = self._initializer_array(value)
                    value.CopyFrom(from_array(array, name=value.name))
        if input_shapes:
            initializers = {value.name for value in model.graph.initializer}
            graph_inputs = {
                value.name: value for value in model.graph.input
                if value.name not in initializers
            }
            for name, shape in input_shapes.items():
                if name not in graph_inputs:
                    raise ValueError(f"Unknown ONNX runtime input: {name}")
                tensor_type = graph_inputs[name].type.tensor_type
                if not tensor_type.HasField("shape") or len(shape) != len(tensor_type.shape.dim):
                    raise ValueError(f"Input shape rank mismatch for {name}")
                for dim, size in zip(tensor_type.shape.dim, shape):
                    if not isinstance(size, Integral) or isinstance(size, bool) or size < 0:
                        raise ValueError(f"Invalid concrete input dimension for {name}: {size}")
                    if dim.HasField("dim_value") and dim.dim_value != size:
                        raise ValueError(f"Input shape contradicts a fixed dimension for {name}")
                    dim.dim_value = int(size)
        if simplify_model is None:
            skip = os.getenv("INFINITENSOR_SKIP_ONNXSIM", "").strip().lower()
            simplify_model = skip not in {"1", "true", "yes"}
        if simplify_model:
            try:
                # The simplifier can mutate its input even if its check fails.
                model_simp, check = simplify(copy.deepcopy(model))
                if check:
                    model = model_simp
            except (ValidationError, RuntimeError):
                pass

        from .onnx_rewrites import fuse_fp32_erf_gelu, fuse_fp32_rms_norm, fuse_masked_scatter
        fuse_masked_scatter(model)
        self.rmsnorm_fusions = fuse_fp32_rms_norm(model) if fuse_fp32_rmsnorm else 0
        self.gelu_fusions = fuse_fp32_erf_gelu(model) if fuse_fp32_gelu else 0

        self.inputs: Dict[str, backend.Tensor] = {}
        self.outputs: Dict[str, backend.Tensor] = {}
        self.tensors: Dict[str, backend.Tensor] = {}
        self.tensor_node_map: Dict[str, str] = {}
        self.initializer: Dict[int, TensorProto] = {}
        self._initializer_by_name: Dict[str, TensorProto] = {}
        self._protected_weight_names = set()
        self.shared_weight_names = set()
        self.shared_weight_bytes = 0
        self.use_naive_allocator: bool = use_naive_allocator
        # try:
        #     model = infer_shapes(model)
        # except:
        #     warnings.warn("infer_shapes failed.")
        self.handler = backend.GraphHandler(runtime)
        from .onnx_layout import INTERNAL_LAYOUT_DOMAIN, lower_conv_channels_last

        if any(node.domain == INTERNAL_LAYOUT_DOMAIN for node in model.graph.node):
            raise ValueError("Reserved internal ONNX layout domain")
        if conv_layout != "default":
            if not self.handler.supports_channels_last_conv():
                raise ValueError("Channels-last Conv requires a compatible runtime")
            lower_conv_channels_last(model.graph, fuse_silu=conv_layout == "channels_last_silu")

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
        conv_bias_views = {}

        def internal_constant(name, values):
            if name in tensors or name in self.tensor_node_map:
                raise ValueError(f"Internal constant name collides: {name}")
            constant = from_array(np.asarray(values), name=name)
            data[name] = constant
            tensor = self.handler.tensor(list(constant.dims), constant.data_type)
            tensor.set_weight()
            tensors[name] = tensor
            return tensor

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

        self._shape_specialized = fold_static_shapes
        if elide_internal_copies and not fold_static_shapes:
            raise ValueError("Eliding internal copies requires a shape-specialized graph")
        public_outputs = {value.name for value in model.graph.output}
        self.elided_internal_copies = 0
        if fold_static_shapes:
            from .static_shape import evaluate_static_shape_node

            if any(not value.type.tensor_type.HasField("shape") or
                   any(not dim.HasField("dim_value") for dim in value.type.tensor_type.shape.dim)
                   for value in model.graph.input if value.name not in data):
                raise ValueError("Static shape folding requires concrete ONNX input shapes")
            opsets = {op.domain: op.version for op in model.opset_import}
        for node_idx in sorted_nodes:
            node = model.graph.node[node_idx]
            if fold_static_shapes:
                value = evaluate_static_shape_node(node, data, tensors, opsets)
                if value is not None:
                    name = node.output[0]
                    constant = from_array(value, name=name)
                    data[name] = constant
                    tensors[name] = self.handler.tensor(list(constant.dims), constant.data_type)
                    tensors[name].set_weight()
                    continue
            if (elide_internal_copies and not node.domain
                    and node.op_type in {"Identity", "Cast", "Reshape"}
                    and node.input[0] not in data and node.output[0] not in public_outputs):
                source = tensors[node.input[0]]
                no_copy = node.op_type == "Identity"
                if node.op_type == "Cast":
                    no_copy = next(a.i for a in node.attribute if a.name == "to") == source.dtype()
                elif node.op_type == "Reshape":
                    shape = _parse_static_input(data, node, 1, required=True)
                    no_copy = list(shape) == source.shape() and all(dim > 0 for dim in shape)
                if no_copy:
                    tensors[node.output[0]] = source
                    self.elided_internal_copies += 1
                    continue
            if node.op_type == "Conv":
                channels_last = node.domain == INTERNAL_LAYOUT_DOMAIN
                source = tensors[node.input[0]]
                weight = tensors[node.input[1]]
                rank = len(source.shape()) - 2
                if rank not in (2, 3) or len(weight.shape()) != rank + 2:
                    raise ValueError("Conv requires rank-4 or rank-5 input and weight")
                attributes = _parse_attribute(
                    node,
                    {
                        "dilations": [1] * rank,
                        "pads": [0] * (2 * rank),
                        "strides": [1] * rank,
                    },
                )
                (d, p, s) = (
                    list(attributes[name]) for name in ("dilations", "pads", "strides")
                )
                if len(d) != rank or len(s) != rank or len(p) != 2 * rank:
                    raise ValueError("Conv attributes must match the spatial rank")
                if rank == 3 and attributes.get("auto_pad", b"NOTSET") != b"NOTSET":
                    raise ValueError("Conv3d requires explicit padding")
                if p[:rank] != p[rank:]:
                    source = self.handler.pad(source, None, p, list(range(-rank, 0)))
                    p = [0] * (2 * rank)
                conv = self.handler.conv if rank == 2 else self.handler.conv3d
                layout = {"channels_last": True} if channels_last else {}
                name = node.output[0]
                output = tensors.get(name)
                bias = tensors[node.input[2]] if len(node.input) > 2 and node.input[2] else None
                if (bias is not None and rank == 2
                        and (bias.dtype() == TensorProto.FLOAT
                             or (fuse_fp16_conv_bias and channels_last
                                 and bias.dtype() == TensorProto.FLOAT16))
                        and self.handler.supports_conv_bias()):
                    tensors[name] = conv(source, weight, output, *p[:rank], *s, *d,
                                         bias=bias, **layout)
                    continue
                value = conv(source, weight, None if bias is not None else output,
                             *p[:rank], *s, *d, **layout)
                if bias is not None:
                    channels = weight.shape()[0]
                    bias_shape = ([1] * (rank + 1) + [channels] if channels_last
                                  else [1, channels] + [1] * rank)
                    bias_name = node.input[2]
                    constant = data.get(bias_name)
                    # Keep FP16 bias addition separate to preserve rounding.
                    if (rank == 2 and constant is not None
                            and constant.data_type == TensorProto.FLOAT16
                            and list(constant.dims) == [channels]):
                        view_key = (bias_name, tuple(bias_shape))
                        if view_key not in conv_bias_views:
                            alias = bias_name + "__conv_bias_view"
                            while alias in known_edge:
                                alias += "_"
                            known_edge.add(alias)
                            view = copy.deepcopy(constant)
                            view.name = alias
                            view.ClearField("dims")
                            view.dims.extend(bias_shape)
                            reshaped = self.handler.tensor(bias_shape, view.data_type)
                            reshaped.set_weight()
                            tensors[alias], data[alias] = reshaped, view
                            conv_bias_views[view_key] = reshaped
                        reshaped = conv_bias_views[view_key]
                    else:
                        reshaped = self.handler.reshape(bias, None, bias_shape)
                    value = self.handler.add(value, reshaped, output)
                tensors[name] = value
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
            elif node.op_type == "Einsum":
                if len(node.input) != 2 or not fold_static_shapes:
                    raise NotImplementedError("Einsum requires two operands and concrete shapes")
                from .einsum import lower_einsum
                equation = _parse_attribute(node)["equation"].decode()
                tensors[node.output[0]] = lower_einsum(
                    self.handler, equation, *(tensors[name] for name in node.input),
                    tensors.get(node.output[0]), backend.ActType.Linear, matmul_compute_type,
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
                attributes = _parse_attribute(node, {"epsilon": 1e-6, "stable_half": 0})
                source, weight = (tensors[name] for name in node.input)
                name = node.output[0]
                epsilon = attributes["epsilon"]
                if attributes["stable_half"]:
                    if (node.domain != "infinitensor" or not source.shape() or
                            backend.tensor_dtype(source) != TensorProto.FLOAT16 or
                            backend.tensor_dtype(weight) != TensorProto.FLOAT16):
                        raise NotImplementedError("Stable half RMSNorm requires InfiniTensor FP16 tensors")
                    if not np.isfinite(epsilon) or epsilon < 0:
                        raise ValueError("RMSNorm epsilon must be finite and nonnegative")
                    one = internal_constant(name + "__rms_one", np.float16(1))
                    eps = internal_constant(name + "__rms_epsilon", np.float16(epsilon))
                    # Scaling before squaring avoids half overflow without an FP32 activation buffer.
                    row_max = self.handler.reduceMax(self.handler.abs(source, None), None, [-1], True)
                    scale = self.handler.max(row_max, one, None)
                    normalized = self.handler.div(source, scale, None)
                    squares = self.handler.mul(normalized, normalized, None)
                    mean = self.handler.reduceMean(squares, None, [-1], True)
                    scaled_eps = self.handler.div(self.handler.div(eps, scale, None), scale, None)
                    inverse = self.handler.rsqrt(self.handler.add(mean, scaled_eps, None), None)
                    result = self.handler.mul(normalized, inverse, None)
                    tensors[name] = self.handler.mul(result, weight, tensors.get(name))
                else:
                    tensors[name] = self.handler.RMSNorm(source, weight, tensors.get(name), epsilon)
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
            elif node.op_type in {"Log", "Sign"}:
                unary = self.handler.log if node.op_type == "Log" else self.handler.sign
                tensors[node.output[0]] = unary(
                    tensors[node.input[0]], tensors.get(node.output[0]),
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
            elif node.op_type == "Not":
                tensors[node.output[0]] = self.handler.logicalNot(
                    tensors[node.input[0]], tensors.get(node.output[0]),
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
                        "nearest_mode": "round_prefer_floor",
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
                source = tensors[node.input[0]]
                shape = source.shape()
                target = list(shape)
                real_axes = list(range(len(shape))) if axes is None else list(axes)
                real_axes = [axis % len(shape) for axis in real_axes] if shape else real_axes
                inverse_scales = {}
                if sizesVal:
                    for axis, size in zip(real_axes, sizesVal):
                        target[axis] = int(size)
                elif scalesVal:
                    for axis, scale in zip(real_axes, scalesVal):
                        if not np.isfinite(scale) or scale <= 0:
                            raise ValueError("Resize scales must be positive and finite")
                        inverse_scales[axis] = 1.0 / scale
                        target[axis] = int(np.floor(shape[axis] * scale))
                linear = mode in ("linear", b"linear")
                coordinates = coordinate_transformation_mode
                if isinstance(coordinates, bytes):
                    coordinates = coordinates.decode()
                singleton = (linear and coordinates in ("half_pixel", "align_corners")
                             and len(shape) == 4 and 1 in target[2:]
                             and bool(sizesVal or scalesVal))
                if attributes["antialias"] or singleton:
                    if not fold_static_shapes:
                        raise NotImplementedError("Antialiased or singleton linear Resize requires concrete input shapes")
                    if (len(shape) != 4 or not linear or not (sizesVal or scalesVal)
                            or target[:2] != shape[:2]
                            or any(inverse_scales.get(i, 1) != 1 for i in (0, 1))):
                        raise NotImplementedError("Antialiased Resize supports static NCHW spatial dimensions")
                    from .resize_weights import linear_resize_weights

                    dtype = {TensorProto.FLOAT: np.float32, TensorProto.FLOAT16: np.float16}.get(
                        backend.tensor_dtype(source)
                    )
                    if dtype is None:
                        raise NotImplementedError("Antialiased Resize supports FP32 and FP16")
                    weights = []
                    for axis in (3, 2):
                        name = f"{node.output[0]}__resize_axis_{axis}"
                        values = linear_resize_weights(
                            shape[axis], target[axis], coordinates,
                            bool(attributes["exclude_outside"]), inverse_scales.get(axis),
                            bool(attributes["antialias"]),
                        ).astype(dtype)
                        weights.append(internal_constant(name, values))
                    width = self.handler.matmul(source, weights[0], None, False, False,
                                                None, backend.ActType.Linear, matmul_compute_type)
                    swapped = self.handler.transpose(width, None, [0, 1, 3, 2])
                    height = self.handler.matmul(swapped, weights[1], None, False, False,
                                                 None, backend.ActType.Linear, matmul_compute_type)
                    tensors[node.output[0]] = self.handler.transpose(height, output, [0, 1, 3, 2])
                    continue
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
            elif node.op_type in {"ReduceL2", "ReduceMax"}:
                attrs = _parse_attribute(node, {"axes": None, "keepdims": 1, "noop_with_empty_axes": 0})
                axes = _parse_static_input(data, node, 1) if _has_input(node, 1) else attrs["axes"]
                input = tensors[node.input[0]]
                output = tensors.get(node.output[0])
                reduce_op = self.handler.reduceL2 if node.op_type == "ReduceL2" else self.handler.reduceMax
                tensors[node.output[0]] = (
                    self.handler.identity(input, output)
                    if not axes and attrs["noop_with_empty_axes"]
                    else reduce_op(input, output, axes or None, bool(attrs["keepdims"]))
                )
            elif node.op_type == "ArgMax":
                attrs = _parse_attribute(node, {"axis": 0, "keepdims": 1, "select_last_index": 0})
                if attrs["select_last_index"]:
                    raise NotImplementedError("ArgMax select_last_index is not supported")
                tensors[node.output[0]] = self.handler.argmax(
                    tensors[node.input[0]], tensors.get(node.output[0]),
                    attrs["axis"], bool(attrs["keepdims"]),
                )
            elif node.op_type == "GridSample":
                attrs = _parse_attribute(node)
                mode = attrs.get("mode", b"linear")
                padding = attrs.get("padding_mode", b"zeros")
                align = attrs.get("align_corners", 0)
                if mode not in {b"linear", b"bilinear", b"nearest"} or padding not in {
                    b"zeros", b"reflection"
                } or align not in (0, 1) or (mode == b"nearest" and padding == b"reflection"):
                    raise NotImplementedError(f"Unsupported GridSample attributes: {attrs}")
                tensors[node.output[0]] = self.handler.gridSample(
                    tensors[node.input[0]], tensors[node.input[1]], tensors.get(node.output[0]),
                    mode == b"nearest", padding == b"reflection", bool(align),
                )
            elif node.op_type == "RoiAlign":
                attrs = _parse_attribute(node)
                coordinates = attrs.get("coordinate_transformation_mode")
                if coordinates is None:
                    from_version = next(op.version for op in model.opset_import if not op.domain)
                    coordinates = b"half_pixel" if from_version >= 16 else b"output_half_pixel"
                if attrs.get("mode", b"avg") != b"avg" or coordinates not in {
                    b"half_pixel", b"output_half_pixel"
                }:
                    raise NotImplementedError(f"Unsupported RoiAlign attributes: {attrs}")
                tensors[node.output[0]] = self.handler.roiAlign(
                    *(tensors[name] for name in node.input), tensors.get(node.output[0]),
                    attrs.get("output_height", 1), attrs.get("output_width", 1),
                    attrs.get("sampling_ratio", 0), attrs.get("spatial_scale", 1.0),
                    coordinates == b"half_pixel",
                )
            elif node.op_type in {"Equal", "Less", "Greater", "GreaterOrEqual", "LessOrEqual"}:
                if node.op_type == "Equal":
                    compare = self.handler.equal
                elif node.op_type in {"Less", "Greater"}:
                    compare = self.handler.less
                else:
                    compare = self.handler.greaterEqual
                lhs, rhs = (tensors[name] for name in node.input)
                if node.op_type in {"Greater", "LessOrEqual"}:
                    lhs, rhs = rhs, lhs
                tensors[node.output[0]] = compare(
                    lhs, rhs, tensors.get(node.output[0])
                )
            elif node.op_type == "Tile":
                repeats = _parse_static_input(data, node, 1, required=True)
                tensors[node.output[0]] = self.handler.tile(
                    tensors[node.input[0]], tensors.get(node.output[0]), repeats,
                )
            elif node.domain == "infinitensor" and node.op_type == "MaskedScatter":
                tensors[node.output[0]] = self.handler.maskedScatter(
                    *(tensors[name] for name in node.input), tensors.get(node.output[0]),
                )
            elif node.op_type == "ScatterElements":
                attrs = _parse_attribute(node)
                reduction = attrs.get("reduction", b"none")
                if reduction not in {b"none", b"add", b"max", b"min"}:
                    raise NotImplementedError(f"Unsupported ScatterElements reduction: {reduction}")
                tensors[node.output[0]] = self.handler.scatterElements(
                    *(tensors[name] for name in node.input), tensors.get(node.output[0]),
                    attrs.get("axis", 0), reduction.decode(),
                )
            elif node.op_type == "ScatterND":
                reduction = _parse_attribute(node).get("reduction", b"none")
                if reduction not in {b"none", b"add"}:
                    raise NotImplementedError(f"Unsupported ScatterND reduction: {reduction}")
                tensors[node.output[0]] = self.handler.scatterNd(
                    *(tensors[name] for name in node.input),
                    tensors.get(node.output[0]), reduction == b"add",
                )
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

                tensors[node.output[0]] = self.handler.slice(
                    tensors[node.input[0]],
                    tensors.get(node.output[0]),
                    clamp(_parse_static_input(data, node, 1, required=True)),
                    clamp(_parse_static_input(data, node, 2, required=True)),
                    (
                        clamp(_parse_static_input(data, node, 3))
                        if _has_input(node, 3)
                        else None
                    ),
                    (
                        clamp(_parse_static_input(data, node, 4))
                        if _has_input(node, 4)
                        else None
                    ),
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
                shape = _parse_static_input(data, node, 1, required=True)
                tensors[node.output[0]] = self.handler.expand(
                    tensors[node.input[0]],
                    tensors.get(node.output[0]),
                    shape,
                )
            elif node.op_type in {"Sin", "Cos"}:
                unary = self.handler.sin if node.op_type == "Sin" else self.handler.cos
                tensors[node.output[0]] = unary(
                    tensors[node.input[0]], tensors.get(node.output[0])
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

        shareable_weights = [
            value.name for value in model.graph.initializer
            if value.name in self._initializer_by_name and value.name not in self.outputs
        ] if weight_store is not None else []
        if weight_store is not None:
            for name in shareable_weights:
                initializer = self._initializer_by_name[name]
                source = weight_store._source(initializer, self._external_data_dir)
                if source is not None:
                    tensors[name].share_weight_data(source)
                    self._protected_weight_names.add(name)
                    self.shared_weight_names.add(name)
                    self.shared_weight_bytes += prod(initializer.dims) * np.dtype(
                        tensor_dtype_to_np_dtype(initializer.data_type)
                    ).itemsize
        self.init()
        if weight_store is not None:
            for name in shareable_weights:
                if name not in self._protected_weight_names and weight_store._remember(
                    self._initializer_by_name[name], self._external_data_dir, tensors[name]
                ):
                    self._protected_weight_names.add(name)

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
                ctx.push_input(it, self._export_initializer(self.initializer.get(it.fuid())))
                for it in op.inputs()
            ]
            outputs = [
                ctx.push_output("{}_{}".format(name, i), it)
                for (i, it) in enumerate(op.outputs())
            ]
            if ty == backend.OpTypeId.Conv:
                ph, pw, dh, dw, sh, sw = backend.conv_attrs_of(op)
                channels_last = backend.conv_channels_last_of(op)
                axis = 3 if channels_last else 1
                group = op.inputs()[0].shape()[axis] // op.inputs()[1].shape()[axis]
                conv_output = outputs
                if channels_last:
                    from .onnx_layout import TO_FIRST, TO_LAST

                    for i in range(2):
                        edge = f"{name}_channels_first_{i}"
                        ctx.push_node(make_node("Transpose", [inputs[i]], [edge], perm=TO_FIRST))
                        inputs[i] = edge
                    conv_output = [name + "_channels_first"]
                ctx.push_node(
                    make_node(
                        ty.name,
                        inputs,
                        conv_output,
                        name,
                        pads=[ph, pw, ph, pw],
                        strides=[sh, sw],
                        dilations=[dh, dw],
                        group=group,
                    )
                )
                if channels_last:
                    ctx.push_node(make_node("Transpose", conv_output, outputs, perm=TO_LAST))
            elif ty == backend.OpTypeId.Silu:
                sigmoid = name + "_sigmoid"
                ctx.push_node(make_node("Sigmoid", inputs, [sigmoid]))
                ctx.push_node(make_node("Mul", [inputs[0], sigmoid], outputs, name))
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

    def init(self) -> None:
        self.handler.data_malloc(self.use_naive_allocator)
        self._copy_initializers()

    def _copy_initializers(self) -> None:
        for name, initializer in self._initializer_by_name.items():
            if name not in self._protected_weight_names:
                self.tensors[name].copyin_numpy(self._initializer_array(initializer))

    def _initializer_array(self, initializer: TensorProto) -> np.ndarray:
        if not uses_external_data(initializer) or self._external_data_dir is None:
            return to_array(initializer)
        array = read_external_array(initializer, self._external_data_dir)
        if array is not None:
            return array
        # ONNX's loader attaches raw_data to its argument; keep the stored
        # initializer metadata lightweight so reinitialization is bounded.
        return to_array(copy.deepcopy(initializer), base_dir=self._external_data_dir)

    def _export_initializer(self, initializer: Optional[TensorProto]) -> Optional[TensorProto]:
        if initializer is not None and self._external_data_dir and uses_external_data(initializer):
            return from_array(self._initializer_array(initializer), name=initializer.name)
        return initializer

    def optimize(self) -> None:
        self.handler.optimize()

    def clone_KV(self, tensor: backend.Tensor) -> backend.Tensor:
        return self.handler.clone_KV(tensor)

    def free_heap(self) -> None:
        self.handler.free_heap()

    def trim_memory(self) -> None:
        self.handler.trim_memory()

    def set_input(self, inputShapes: List[Sequence[int]]) -> None:
        if len(inputShapes) != len(self.inputs):
            raise ValueError(
                "inputShapes must contain one shape per model input; expected "
                "{}, got {}".format(len(self.inputs), len(inputShapes))
            )
        if self._shape_specialized and any(
            list(new_shape) != list(tensor.shape())
            for new_shape, tensor in zip(inputShapes, self.inputs.values())
        ):
            raise ValueError("Reimport the ONNX model to change specialized input shapes")
        for newInput, oldInput in zip(inputShapes, self.inputs):
            oldTensor = self.inputs[oldInput]
            self.handler.change_shape(newInput, oldTensor.fuid())
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

    def run_with_graph(self) -> None:
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
