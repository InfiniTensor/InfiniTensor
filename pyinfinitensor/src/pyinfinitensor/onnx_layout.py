"""Lower Conv2D chains to packed channels-last tensors."""

import copy
from collections import Counter

import numpy as np
from onnx import TensorProto
from onnx.external_data_helper import uses_external_data
from onnx.helper import get_attribute_value, make_attribute, make_node
from onnx.numpy_helper import from_array, to_array


INTERNAL_LAYOUT_DOMAIN = "infinitensor.internal.layout"
TO_LAST = [0, 2, 3, 1]
TO_FIRST = [0, 3, 1, 2]


def lower_conv_channels_last(graph, fuse_silu=False):
    inputs = {x.name for x in graph.input}
    outputs = {x.name for x in graph.output}
    weights = {x.name: x for x in graph.initializer if x.name not in inputs}
    nodes = list(graph.node)
    names = inputs | outputs | {x.name for x in graph.initializer}
    names.update(x.name for x in graph.value_info)
    names.update(edge for node in nodes for edge in (*node.input, *node.output))
    uses = Counter(edge for node in nodes for edge in node.input)
    producer = {out: i for i, node in enumerate(nodes) for out in node.output}

    def fresh(name):
        while name in names:
            name += "_"
        names.add(name)
        return name

    def is_conv(node):
        if node.domain or node.op_type != "Conv" or len(node.input) not in (2, 3):
            return False
        weight = weights.get(node.input[1])
        attrs = {a.name: get_attribute_value(a) for a in node.attribute}
        pads = attrs.get("pads", [0, 0, 0, 0])
        return (
            weight is not None and not uses_external_data(weight)
            and len(weight.dims) == 4 and weight.data_type in (TensorProto.FLOAT, TensorProto.FLOAT16)
            and len(pads) == 4 and pads[:2] == pads[2:]
            and attrs.get("auto_pad", b"NOTSET") == b"NOTSET"
        )

    convs = {i for i, node in enumerate(nodes) if is_conv(node)}
    conv_outputs = {nodes[i].output[0] for i in convs}
    if not convs:
        return
    physical = {name: fresh(name + "__nhwc") for name in sorted(conv_outputs)}
    silus, sigmoids = {}, set()
    for i, node in enumerate(nodes):
        if node.domain or node.op_type != "Mul" or len(node.input) != 2:
            continue
        for value, sigmoid in (tuple(node.input), tuple(reversed(node.input))):
            if value not in conv_outputs or sigmoid not in producer:
                continue
            index = producer[sigmoid]
            src = nodes[index]
            if (not src.domain and src.op_type == "Sigmoid"
                    and list(src.input) == [value]
                    and uses[sigmoid] == 1 and sigmoid not in outputs):
                silus[i] = (value, sigmoid)
                sigmoids.add(index)
                physical[node.output[0]] = fresh(node.output[0] + "__nhwc")
                if not fuse_silu:
                    physical[sigmoid] = fresh(sigmoid + "__nhwc")
                break

    # Only propagate through operators whose semantics do not depend on memory
    # order, or whose channel axis can be translated explicitly.
    propagated = {}
    changed = True
    while changed:
        changed = False
        for i, node in enumerate(nodes):
            if (node.domain or i in convs or i in silus
                    or i in sigmoids or i in propagated):
                continue
            op = node.op_type
            if op in {"Add", "Mul", "Concat"}:
                edges = list(node.input)
            elif op in {"Split", "Sigmoid", "Relu", "Tanh", "Identity"}:
                edges = [node.input[0]]
            else:
                continue
            if not edges or any(name not in physical for name in edges):
                continue
            axis = None
            if op in {"Concat", "Split"}:
                axis = next((a.i for a in node.attribute if a.name == "axis"), 0)
                if not -4 <= axis < 4:
                    continue
                axis = TO_LAST.index(axis % 4)
            propagated[i] = (len(edges), axis)
            for name in node.output:
                physical[name] = fresh(name + "__nhwc")
            changed = True

    rewritten, packed, input_views, public_views = [], {}, {}, set()

    def channels_first(name):
        if name in physical and name not in public_views:
            rewritten.append(make_node("Transpose", [physical[name]], [name], perm=TO_FIRST))
            public_views.add(name)
        return name

    def channels_last(name):
        if name in physical:
            return physical[name]
        if name not in input_views:
            alias = fresh(name + "__nhwc")
            rewritten.append(make_node("Transpose", [name], [alias], perm=TO_LAST))
            input_views[name] = alias
        return input_views[name]

    for i, original in enumerate(nodes):
        node = copy.deepcopy(original)
        if i in convs:
            weight = node.input[1]
            if weight not in packed:
                alias = fresh(weight + "__ohwi")
                graph.initializer.append(from_array(
                    np.ascontiguousarray(to_array(weights[weight]).transpose(TO_LAST)), alias
                ))
                packed[weight] = alias
            node.input[0] = channels_last(node.input[0])
            node.input[1] = packed[weight]
            node.output[0] = physical[node.output[0]]
            node.domain = INTERNAL_LAYOUT_DOMAIN
            if len(node.input) == 3 and node.input[2]:
                node.input[2] = channels_first(node.input[2])
        elif i in silus:
            value, sigmoid = silus[i]
            out = physical[node.output[0]]
            if fuse_silu:
                node = make_node("Silu", [physical[value]], [out], name=node.name)
            else:
                node.input[:] = [physical[name] for name in original.input]
                node.output[0] = out
        elif i in sigmoids:
            if fuse_silu:
                continue
            node.input[0] = physical[node.input[0]]
            node.output[0] = physical[node.output[0]]
        elif i in propagated:
            count, axis = propagated[i]
            node.input[:] = [physical[name] for name in original.input[:count]] + [
                channels_first(name) for name in original.input[count:]
            ]
            node.output[:] = [physical[name] for name in original.output]
            if axis is not None:
                for attr in node.attribute:
                    if attr.name == "axis":
                        attr.i = axis
                        break
                else:
                    node.attribute.append(make_attribute("axis", axis))
        else:
            node.input[:] = [channels_first(name) for name in original.input]
        rewritten.append(node)
    for name in sorted(outputs):
        channels_first(name)
    for name in packed:
        if name not in outputs and not any(name in node.input for node in rewritten):
            for i, weight in enumerate(graph.initializer):
                if weight.name == name:
                    del graph.initializer[i]
                    break
    graph.ClearField("node")
    graph.node.extend(rewritten)
