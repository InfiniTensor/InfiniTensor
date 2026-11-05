"""Recognize exporter patterns without specializing runtime input values."""

from collections import Counter
from math import isfinite

from onnx import AttributeProto, TensorProto, helper, numpy_helper
from onnx.shape_inference import InferenceError, infer_shapes


def fuse_fp32_erf_gelu(model):
    """Fold FP32 x * (1 + erf(x / sqrt(2))) * 0.5 when intermediates are private."""
    graph = model.graph
    if not any(node.op_type == "Erf" for node in graph.node):
        return 0
    if any(attr.type in (AttributeProto.GRAPH, AttributeProto.GRAPHS)
           for node in graph.node for attr in node.attribute):
        return 0
    nodes = {node.output[0]: node for node in graph.node if len(node.output) == 1}
    public = {value.name for value in graph.output}
    inputs = {value.name for value in graph.input}
    constants = {value.name: value for value in graph.initializer if value.name not in inputs}
    uses = Counter(edge for node in graph.node for edge in node.input)

    def find(name, kind, arity):
        node = nodes.get(name)
        if node is not None and not node.domain and node.op_type == kind and len(node.input) == arity:
            return node
        return None

    def scalar(name, expected):
        tensor = constants.get(name)
        if tensor is None and (node := find(name, "Constant", 0)) is not None:
            tensor = next((attr.t for attr in node.attribute if attr.name == "value"), None)
        if (tensor is None or tensor.data_type != TensorProto.FLOAT
                or tensor.data_location == TensorProto.EXTERNAL or tensor.dims):
            return False
        value = numpy_helper.to_array(tensor)
        return value.item() == value.dtype.type(expected)

    def commuted(edges):
        return (tuple(edges), tuple(reversed(edges)))

    def without_scalar(node, value):
        if node is not None:
            return next((x for x, constant in commuted(node.input) if scalar(constant, value)), None)
        return None

    replace, remove = {}, set()
    for name, node in nodes.items():
        if find(name, "Mul", 2) is None:
            continue
        choices = []
        product_name = without_scalar(node, .5)
        product = find(product_name, "Mul", 2)
        if product is not None:
            choices.extend((x, gate, product_name) for x, gate in commuted(product.input))
        for half, gate in commuted(node.input):
            x = without_scalar(find(half, "Mul", 2), .5)
            if x is not None:
                choices.append((x, gate, half))
        for source, gate, outer in choices:
            erf_name = without_scalar(find(gate, "Add", 2), 1.)
            erf = find(erf_name, "Erf", 1)
            if erf is None:
                continue
            normalized = erf.input[0]
            div = find(normalized, "Div", 2)
            if not ((div is not None and div.input[0] == source and scalar(div.input[1], 2**.5))
                    or without_scalar(find(normalized, "Mul", 2), 2**-.5) == source):
                continue
            internal = {outer, gate, erf_name, normalized}
            if any(edge in public or uses[edge] != 1 or edge in remove or edge in replace
                   for edge in internal):
                continue
            replace[name] = helper.make_node("Gelu", [source], [name], name=node.name,
                                             domain="infinitensor")
            remove.update(internal)
            break
    if replace:
        kept = [replace.get(node.output[0], node) if node.output else node
                for node in graph.node if not node.output or node.output[0] not in remove]
        del graph.node[:]
        graph.node.extend(kept)
        if not any(entry.domain == "infinitensor" for entry in model.opset_import):
            model.opset_import.append(helper.make_opsetid("infinitensor", 1))
    return len(replace)


def fuse_fp32_rms_norm(model):
    """Fold x * reciprocal(sqrt(mean(x**2) + eps)) * weight on the last axis."""
    graph = model.graph
    if not any(node.op_type == "ReduceMean" for node in graph.node):
        return 0
    if any(attr.type in (AttributeProto.GRAPH, AttributeProto.GRAPHS)
           for node in graph.node for attr in node.attribute):
        return 0
    try:
        inferred = infer_shapes(model).graph
    except (InferenceError, RuntimeError):
        return 0
    types = {value.name: value.type.tensor_type for value in
             (*inferred.input, *inferred.value_info, *inferred.output)}
    inputs = {value.name for value in graph.input}
    outputs = {value.name for value in graph.output}
    initializers = {x.name: x for x in graph.initializer if x.name not in inputs}
    nodes = {node.output[0]: node for node in graph.node if len(node.output) == 1}
    uses = Counter(edge for node in graph.node for edge in node.input)

    def attrs(node):
        return {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}

    def find(edge, op, arity=None):
        node = nodes.get(edge)
        if (node is not None and not node.domain and node.op_type == op
                and (arity is None or len(node.input) == arity)):
            return node
        return None

    def constant(edge):
        tensor = initializers.get(edge)
        if tensor is None and (node := find(edge, "Constant", 0)) is not None:
            tensor = attrs(node).get("value")
        if isinstance(tensor, TensorProto) and tensor.data_location != TensorProto.EXTERNAL:
            return numpy_helper.to_array(tensor)
        return None

    def scalar(edge):
        array = constant(edge)
        if array is not None and array.dtype.name == "float32" and array.ndim == 0:
            return float(array)
        return None

    def both(edges):
        return (edges, reversed(edges))

    replace, remove = {}, set()
    for edge in nodes:
        final = find(edge, "Mul", 2)
        if final is None:
            continue
        for normalized, scale in both(final.input):
            weight = initializers.get(scale)
            if weight is None or weight.data_type != TensorProto.FLOAT or len(weight.dims) != 1:
                continue
            cast = find(normalized, "Cast", 1)
            if cast is not None:
                if attrs(cast).get("to") != TensorProto.FLOAT:
                    continue
                normalized = cast.input[0]
            norm = find(normalized, "Mul", 2)
            if norm is None:
                continue
            for x, inverse in both(norm.input):
                source = types.get(x)
                if (source is None or source.elem_type != TensorProto.FLOAT or not source.shape.dim
                        or source.shape.dim[-1].dim_value != weight.dims[0] or weight.dims[0] < 1):
                    continue
                div = find(inverse, "Div", 2)
                if div is None or scalar(div.input[0]) != 1:
                    continue
                sqrt = find(div.input[1], "Sqrt", 1)
                add = find(sqrt.input[0], "Add", 2) if sqrt is not None else None
                if add is None:
                    continue
                for average, epsilon in both(add.input):
                    mean, eps = find(average, "ReduceMean"), scalar(epsilon)
                    if mean is None or eps is None or not isfinite(eps) or eps < 0:
                        continue
                    options = attrs(mean)
                    axes = options.get("axes")
                    if len(mean.input) == 2:
                        axes = constant(mean.input[1])
                        axes = axes.tolist() if axes is not None and axes.ndim == 1 else None
                    if axes not in ([-1], [len(source.shape.dim) - 1]) or options.get("keepdims", 1) != 1:
                        continue
                    square = find(mean.input[0], "Pow", 2)
                    if square is None or square.input[0] != x or scalar(square.input[1]) != 2:
                        continue
                    chain = [square, mean, add, sqrt, div, norm] + ([cast] if cast is not None else [])
                    internal = {node.output[0] for node in chain}
                    if any(name in outputs or uses[name] != 1 or name in replace or name in remove
                           for name in internal):
                        continue
                    replace[edge] = helper.make_node("RMSNorm", [x, scale], [edge], name=final.name,
                                                     domain="infinitensor", epsilon=eps)
                    remove.update(internal)
                    break
                if edge in replace:
                    break
            if edge in replace:
                break
    if replace:
        kept = [replace.get(node.output[0], node) if node.output else node
                for node in graph.node if not node.output or node.output[0] not in remove]
        del graph.node[:]
        graph.node.extend(kept)
        if not any(entry.domain == "infinitensor" for entry in model.opset_import):
            model.opset_import.append(helper.make_opsetid("infinitensor", 1))
    return len(replace)


def fuse_masked_scatter(model):
    graph = model.graph
    if any(attr.type in (AttributeProto.GRAPH, AttributeProto.GRAPHS)
           for node in graph.node for attr in node.attribute):
        return 0
    producers = {name: node for node in graph.node for name in node.output if name}
    initializers = {value.name: value for value in graph.initializer}

    def attributes(node):
        return {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}

    def find(name, kind, arity=None):
        node = producers.get(name)
        if (node is not None and node.domain in ("", "ai.onnx") and node.op_type == kind
                and (arity is None or len(node.input) == arity)):
            return node
        return None

    def integer_constant(name, value, scalar=False):
        tensor = initializers.get(name)
        if tensor is None:
            node = find(name, "Constant", 0)
            tensor = attributes(node).get("value") if node is not None else None
        if (not isinstance(tensor, TensorProto) or tensor.data_type not in
                (TensorProto.INT64, TensorProto.INT32) or tensor.dims not in ([], [1])):
            return False
        if scalar and tensor.dims:
            return False
        if tensor.data_location == TensorProto.EXTERNAL:
            return False
        return int(numpy_helper.to_array(tensor).item()) == value

    replacements = {}
    matched = set()
    for scatter in graph.node:
        if (scatter.domain not in ("", "ai.onnx") or scatter.op_type != "ScatterND"
                or len(scatter.input) != 3 or len(scatter.output) != 1
                or attributes(scatter).get("reduction", b"none") != b"none"):
            continue
        data, coordinates, updates = scatter.input
        transpose = find(coordinates, "Transpose", 1)
        if transpose is None or attributes(transpose).get("perm", [1, 0]) != [1, 0]:
            continue
        nonzero = find(transpose.input[0], "NonZero", 1)
        sliced = find(updates, "Slice")
        if nonzero is None or sliced is None or len(sliced.input) not in (4, 5):
            continue
        flat = find(sliced.input[0], "Reshape", 2)
        if flat is None or not integer_constant(flat.input[1], -1):
            continue
        if (not integer_constant(sliced.input[1], 0)
                or not integer_constant(sliced.input[3], 0)
                or (len(sliced.input) == 5 and not integer_constant(sliced.input[4], 1))):
            continue
        end = find(sliced.input[2], "Unsqueeze", 2)
        if end is None or not integer_constant(end.input[1], 0):
            continue
        gather = find(end.input[0], "Gather", 2)
        if (gather is None or attributes(gather).get("axis", 0) != 0
                or not integer_constant(gather.input[1], 0, scalar=True)):
            continue
        shape = find(gather.input[0], "Shape", 1)
        if shape is None or shape.attribute or shape.input[0] != coordinates:
            continue
        mask = nonzero.input[0]
        # Remove only expands whose target is exactly the destination shape.
        while (expand := find(mask, "Expand", 2)) is not None:
            target = find(expand.input[1], "Shape", 1)
            if target is None or target.attribute or target.input[0] != data:
                break
            mask = expand.input[0]
        replacements[scatter.output[0]] = helper.make_node(
            "MaskedScatter", [data, mask, flat.input[0]], list(scatter.output),
            name=scatter.name, domain="infinitensor",
        )
        matched.update((coordinates, updates))

    if not replacements:
        return 0

    # Drop only the now-unused input subgraphs of replaced ScatterND nodes.
    candidates = set()
    while matched:
        edge = matched.pop()
        if edge in candidates or edge not in producers:
            continue
        candidates.add(edge)
        matched.update(name for name in producers[edge].input if name)
    nodes = [replacements.get(node.output[0], node) if node.output else node
             for node in graph.node]
    live = {value.name for value in graph.output}
    for node in reversed(nodes):
        if not node.output or any(name not in candidates for name in node.output):
            live.update(node.input)
    nodes_by_output = {name: node for node in nodes for name in node.output}
    pending = list(live)
    while pending:
        node = nodes_by_output.get(pending.pop())
        if node is None:
            continue
        for name in node.input:
            if name and name not in live:
                live.add(name)
                pending.append(name)
    kept = [node for node in nodes if not node.output or any(
        name not in candidates or name in live for name in node.output
    )]
    del graph.node[:]
    graph.node.extend(kept)
    if not any(entry.domain == "infinitensor" for entry in model.opset_import):
        model.opset_import.append(helper.make_opsetid("infinitensor", 1))
    return len(replacements)
