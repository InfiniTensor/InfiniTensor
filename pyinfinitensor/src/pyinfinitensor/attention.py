"""Preserve packed token projections while separating independent attention segments."""

import copy
from collections import Counter

import numpy as np
from onnx import helper, numpy_helper, shape_inference


def segment_attention(model, lengths, *, expected_count=None):
    """Return an ONNX copy with unmasked QK-softmax-V attention split by length.

    The input graph must use rank-four [batch, heads, tokens, channels]
    attention and have no attention mask. All projections and MLPs remain
    packed. No learned parameters are changed and no host computation is added.
    """
    lengths = tuple(lengths)
    if any(not isinstance(length, (int, np.integer)) or isinstance(length, bool) for length in lengths):
        raise ValueError("Attention segment lengths must be integers")
    lengths = tuple(int(length) for length in lengths)
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("Attention segment lengths must be positive")
    result = copy.deepcopy(model)
    if len(lengths) == 1:
        return result
    nodes = list(result.graph.node)
    producers = {output: node for node in nodes for output in node.output}
    shaped = shape_inference.infer_shapes(result)
    ranks = {value.name: len(value.type.tensor_type.shape.dim)
             for value in [*shaped.graph.input, *shaped.graph.value_info, *shaped.graph.output]}
    shapes = {value.name: value.type.tensor_type.shape.dim
              for value in [*shaped.graph.input, *shaped.graph.value_info, *shaped.graph.output]}
    users = Counter(value for node in nodes for value in node.input)
    public = {value.name for value in result.graph.output}
    replacements, removed = {}, set()
    for final in nodes:
        if final.op_type != "MatMul":
            continue
        value = final.input[0]
        chain, found_softmax = [], False
        while value in producers:
            node = producers[value]
            if node.op_type == "Cast":
                previous = node.input[0]
            elif node.op_type == "Softmax":
                axis = next((attr.i for attr in node.attribute if attr.name == "axis"), -1)
                if found_softmax or axis not in {-1, 3}:
                    break
                found_softmax = True
                previous = node.input[0]
            elif node.op_type == "Mul" and found_softmax:
                candidates = [item for item in node.input if item in producers
                              and producers[item].op_type in {"MatMul", "Cast"}]
                if len(candidates) != 1:
                    break
                previous = candidates[0]
            elif node.op_type == "MatMul" and found_softmax:
                chain.append(node)
                break
            else:
                break
            chain.append(node)
            value = previous
        if not chain or chain[-1].op_type != "MatMul" or not found_softmax:
            continue
        if any(users[node.output[0]] != 1 or node.output[0] in public for node in chain):
            continue
        first = chain[-1]
        if any(ranks.get(name) != 4 for name in [*first.input, final.input[1]]):
            continue
        for name, axis in [(first.input[0], 2), (first.input[1], 3), (final.input[1], 2)]:
            dimension = shapes[name][axis]
            if dimension.HasField("dim_value") and dimension.dim_value != sum(lengths):
                raise ValueError("Segment lengths do not cover the attention sequence")
        chain.reverse()
        prefix = "__segmented_attention_" + str(nodes.index(final))
        generated, outputs, start = [], [], 0
        for index, length in enumerate(lengths):
            stem = prefix + "_" + str(index)
            mapping = {}
            for label, source, axis in [("q", first.input[0], 2), ("kt", first.input[1], 3),
                                         ("v", final.input[1], 2)]:
                arguments = []
                for key, values in [("start", [start]), ("end", [start + length]), ("axis", [axis])]:
                    name = stem + "_" + label + "_" + key
                    result.graph.initializer.append(numpy_helper.from_array(np.asarray(values, np.int64), name))
                    arguments.append(name)
                sliced = stem + "_" + label
                generated.append(helper.make_node("Slice", [source, *arguments], [sliced], name=sliced))
                mapping[source] = sliced
            for operation_index, node in enumerate([*chain, final]):
                clone = copy.deepcopy(node)
                clone.name = stem + "_op_" + str(operation_index)
                clone.input[:] = [mapping.get(item, item) for item in node.input]
                for i, output in enumerate(node.output):
                    mapping[output] = stem + "_value_" + str(len(mapping))
                    clone.output[i] = mapping[output]
                generated.append(clone)
            outputs.append(mapping[final.output[0]])
            start += length
        generated.append(helper.make_node("Concat", outputs, list(final.output), axis=2,
                                          name=prefix + "_concat"))
        replacements[id(final)] = generated
        removed.update(id(node) for node in [*chain, final])
    if not replacements:
        raise ValueError("No private unmasked rank-four attention pattern found")
    if expected_count is not None and len(replacements) != expected_count:
        raise ValueError(
            f"Expected {expected_count} independent attention blocks, found {len(replacements)}"
        )
    rewritten = []
    for node in nodes:
        if id(node) in replacements:
            rewritten.extend(replacements[id(node)])
        elif id(node) not in removed:
            rewritten.append(node)
    del result.graph.node[:]
    result.graph.node.extend(rewritten)
    # Old inferred attention shapes no longer describe the segmented graph.
    del result.graph.value_info[:]
    return result
