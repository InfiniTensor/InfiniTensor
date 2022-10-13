from pyinfinitensor import *

import functools

import numpy as np

import onnx
import onnx.checker
import onnx.numpy_helper
import onnx.shape_inference

def _add_value_info_for_constants(model : onnx.ModelProto):
    """
    Currently onnx.shape_inference doesn't use the shape of initializers, so add
    that info explicitly as ValueInfoProtos.
    Mutates the model.
    Args:
        model: The ModelProto to update.
    """
    # All (top-level) constants will have ValueInfos before IRv4 as they are all inputs
    if model.ir_version < 4:
        return

    def add_const_value_infos_to_graph(graph : onnx.GraphProto):
        inputs = {i.name for i in graph.input}
        existing_info = {vi.name: vi for vi in graph.value_info}
        for init in graph.initializer:
            # Check it really is a constant, not an input
            if init.name in inputs:
                continue

            # The details we want to add
            elem_type = init.data_type
            shape = init.dims

            # Get existing or create new value info for this constant
            vi = existing_info.get(init.name)
            if vi is None:
                vi = graph.value_info.add()
                vi.name = init.name

            # Even though it would be weird, we will not overwrite info even if it doesn't match
            tt = vi.type.tensor_type
            if tt.elem_type == onnx.TensorProto.UNDEFINED:
                tt.elem_type = elem_type
            if not tt.HasField("shape"):
                # Ensure we set an empty list if the const is scalar (zero dims)
                tt.shape.dim.extend([])
                for dim in shape:
                    tt.shape.dim.add().dim_value = dim

        # Handle subgraphs
        for node in graph.node:
            for attr in node.attribute:
                # Ref attrs refer to other attrs, so we don't need to do anything
                if attr.ref_attr_name != "":
                    continue

                if attr.type == onnx.AttributeProto.GRAPH:
                    add_const_value_infos_to_graph(attr.g)
                if attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        add_const_value_infos_to_graph(g)

    return add_const_value_infos_to_graph(model.graph)


def _parse_attribute(attributes, defaults=dict()):
    atts = defaults
    for att in attributes:
        if att.type == onnx.AttributeProto.INT:
            atts[att.name] = att.i
        elif att.type == onnx.AttributeProto.INTS:
            atts[att.name] = att.ints
        elif att.type == onnx.AttributeProto.FLOAT:
            atts[att.name] = att.f
        elif att.type == onnx.AttributeProto.STRING:
            atts[att.name] = att.s
        elif att.type == onnx.AttributeProto.TENSOR:
            atts[att.name] = att.t
        else:
            assert False, "Unsupported Attribute Type: {}".format(att.type)
    return atts


def _onnx_datatype_tostring(dtype):
    if dtype == 0:
        return 'UNDEFINED'
    elif dtype == 1:
        return 'FLOAT'
    elif dtype == 2:
        return 'UINT8'
    elif dtype == 3:
        return 'INT8'
    elif dtype == 4:
        return 'UINT16'
    elif dtype == 5:
        return 'INT16'
    elif dtype == 6:
        return 'INT32'
    elif dtype == 7:
        return 'INT64'
    elif dtype == 8:
        return 'STRING'
    elif dtype == 9:
        return 'BOOL'
    elif dtype == 10:
        return 'FLOAT16'
    elif dtype == 11:
        return 'DOUBLE'
    elif dtype == 12:
        return 'UINT32'
    elif dtype == 13:
        return 'UINT64'
    elif dtype == 14:
        return 'COMPLEX64'
    elif dtype == 15:
        return 'COMPLEX128'
    elif dtype == 16:
        return 'BFLOAT16'
    else:
        assert False, 'Unknown onnx datatype'


def import_onnx(gf: GraphFactory, net: str):
    ts, ds, ops, consts = dict(), dict(), dict(), dict() # (key, value) = (name, class)
    model = onnx.load(net)

    # Tensor_input
    for input in model.graph.input:
        if input.name not in ts:
            dims = [d.dim_value for d in input.type.tensor_type.shape.dim]
            ts[input.name] = gf.g.addTensor(dims, _onnx_datatype_tostring(input.type.tensor_type.elem_type))
            ds[input.name] = dims

    # Tensor_weight
    for weight in model.graph.initializer:
        if weight.name not in ts:
            ts[weight.name] = gf.g.addTensor(weight.dims, _onnx_datatype_tostring(weight.data_type))
            ds[weight.name] = weight.dims

    # Tensor_inference
    _add_value_info_for_constants(model)
    infered_model = onnx.shape_inference.infer_shapes(model)
    for v in infered_model.graph.value_info:
        if v.name not in ts:
            dims = [d.dim_value for d in v.type.tensor_type.shape.dim]
            ts[v.name] = gf.g.addTensor(dims, _onnx_datatype_tostring(v.type.tensor_type.elem_type))
            ds[v.name] = dims

    # Tensor_output
    for output in model.graph.output:
        if output.name not in ts:
            dims = [d.dim_value for d in output.type.tensor_type.shape.dim]
            ts[output.name] = gf.g.addTensor(dims, _onnx_datatype_tostring(output.type.tensor_type.elem_type))
            ds[output.name] = dims

    # Op
    for node in model.graph.node:
        if node.op_type == 'Conv':
            attrs = _parse_attribute(node.attribute, {
                "auto_pad": "NOTSET",
                "dilations": [1, 1],
                "pads": [0, 0, 0, 0],
                "strides": [1, 1]})
            assert len(node.input) == 2 or len(node.input) == 3
            assert len(node.output) == 1
            assert attrs["auto_pad"] == "NOTSET"
            assert len(attrs["pads"]) == 4
            assert len(attrs["strides"]) == 2
            assert len(attrs["dilations"]) == 2
            assert attrs["pads"][0] == attrs["pads"][2]
            assert attrs["pads"][1] == attrs["pads"][3]
            gf.conv(ts[node.input[0]], ts[node.input[1]], ts[node.output[0]],
                    attrs["pads"][0], attrs["pads"][1],
                    attrs["strides"][0], attrs["strides"][1],
                    attrs["dilations"][0], attrs["dilations"][1],
                    None if len(node.input) == 2 else ts[node.input[2]])

        # elif node.op_type == 'MatMul':
        #     assert len(node.input) == 2
        #     assert len(node.output) == 1
        #     dimA = list(ds[node.input[0]])
        #     dimB = list(ds[node.input[1]])
        #     dimO = list(ds[node.output[0]])

        #     if len(dimA) == 2 and len(dimB) == 2:
        #         tmpI0 = g.tensor([1] + list(ds[node.input[0]]), "FLOAT")
        #         tmpI1 = g.tensor([1] + list(ds[node.input[1]]), "FLOAT")
        #         tmpO = g.tensor([1] + list(ds[node.output[0]]), "FLOAT")
        #         gf.transpose(ts[node.input[0]], tmpI0, 0, Perm([PermItem(-1), PermItem(0), PermItem(1)]), 1)
        #         gf.transpose(ts[node.input[1]], tmpI1, 0, Perm([PermItem(-1), PermItem(0), PermItem(1)]), 1)
        #         gf.matmul(tmpI0, tmpI1, tmpO, False, False, None)
        #         gf.transpose(tmpO, ts[node.output[0]], -1, Perm([PermItem([0, 1]), PermItem(2)]), 0)

        #     else:
        #         assert len(dimO) >= 3
        #         batch = functools.reduce(lambda x, y: x * y, dimO[:-2])

        #         if len(dimA) == 3:
        #             tmpI0 = ts[node.input[0]]
        #         else:
        #             tmpI0 = g.tensor([batch, dimA[-2], dimA[-1]], "FLOAT")
        #             gf.reshape(ts[node.input[0]], tmpI0)

        #         if len(dimB) == 3:
        #             tmpI1 = ts[node.input[1]]
        #         else:
        #             tmpI1 = g.tensor([batch, dimB[-2], dimB[-1]], "FLOAT")
        #             gf.reshape(ts[node.input[1]], tmpI1)

        #         if len(dimO) == 3:
        #             tmpO = ts[node.output[0]]
        #             gf.matmul(tmpI0, tmpI1, tmpO, False, False, None)
        #         else:
        #             tmpO = g.tensor([batch, dimO[-2], dimO[-1]], "FLOAT")
        #             gf.matmul(tmpI0, tmpI1, tmpO, False, False, None)
        #             gf.reshape(tmpO, ts[node.output[0]])

        # elif node.op_type == 'ConvTranspose':
        #     attrs = _parse_attribute(node.attribute, {
        #         "auto_pad": "NOTSET",
        #         "dilations": [1, 1],
        #         "pads": [0, 0, 0, 0],
        #         "strides": [1, 1]})
        #     assert len(node.input) == 2 or len(node.input) == 3
        #     assert len(node.output) == 1
        #     assert attrs["auto_pad"] == "NOTSET"
        #     assert len(attrs["pads"]) == 4
        #     assert len(attrs["strides"]) == 2
        #     assert len(attrs["dilations"]) == 2
        #     assert attrs["pads"][0] == attrs["pads"][2]
        #     assert attrs["pads"][1] == attrs["pads"][3]
        #     oph, opw = 0, 0
        #     if "output_padding" in attrs:
        #         oph, opw = attrs["output_padding"][0], attrs["output_padding"][1]
        #         assert attrs["output_padding"][0] == attrs["output_padding"][1]
        #     group = 0
        #     if "group" in attrs:
        #         group = attrs["group"][0]
        #     gf.convTrans(ts[node.input[0]], ts[node.input[1]], ts[node.output[0]],
        #                 attrs["pads"][0], attrs["pads"][1],
        #                 attrs["strides"][0], attrs["strides"][1],
        #                 attrs["dilations"][0], attrs["dilations"][1],
        #                 oph, opw, group,
        #                 None if len(node.input) == 2 else ts[node.input[2]])
        
        # elif node.op_type == 'Pad':
        #     attrs = _parse_attribute(node.attribute, {'mode': b'constant'})
        #     assert attrs["mode"].decode("ascii") == "constant"
        #     assert len(attrs["pads"]) % 2 == 0
        #     assert attrs["value"] == 0
        #     nDim = len(attrs["pads"]) // 2
        #     begin = attrs["pads"][:nDim]
        #     end = attrs["pads"][nDim:]
        #     g.pad(ts[node.input[0]], ts[node.output[0]], begin, end)

        # elif node.op_type == "Slice":
        #     attrs = _parse_attribute(node.attribute, {})
        #     assert len(node.input) == 4
        #     assert len(node.output) == 1
        #     g.slice(ts[node.input[0]], ts[node.output[0]],
        #             ts[node.input[1]], ts[node.input[2]])

        # elif node.op_type == 'Concat':
        #     assert len(node.output) == 1
        #     attrs = _parse_attribute(node.attribute, {})
        #     g.concat([ts[item] for item in node.input], ts[node.output[0]], attrs["axis"])

        # elif node.op_type == "Split":
        #     attrs = _parse_attribute(node.attribute, {})
        #     assert len(node.input) == 1 
        #     assert len(node.output) > 1
        #     axis = attrs['axis']
        #     split = attrs['split']
        #     g.split(ts[node.input[0]], [ts[t] for t in node.output], axis, split)

        # elif node.op_type == 'MaxPool':
        #     attrs = _parse_attribute(node.attribute, {
        #         "auto_pad": "NOTSET",
        #         "dilations": [1, 1],
        #         "pads": [0, 0, 0, 0],
        #         "strides": [1, 1]})
        #     assert len(node.input) == 1
        #     assert len(node.output) == 1
        #     assert len(attrs["kernel_shape"]) == 2
        #     assert len(attrs["pads"]) == 4
        #     assert len(attrs["strides"]) == 2
        #     assert len(attrs["dilations"]) == 2
        #     assert attrs["pads"][0] == attrs["pads"][2]
        #     assert attrs["pads"][1] == attrs["pads"][3]
        #     gf.maxpool(ts[node.input[0]], ts[node.output[0]],
        #             attrs["kernel_shape"][0], attrs["kernel_shape"][1],
        #             attrs["dilations"][0], attrs["dilations"][1],
        #             attrs["pads"][0], attrs["pads"][1],
        #             attrs["strides"][0], attrs["strides"][1])

        # elif node.op_type == 'AveragePool':
        #     attrs = _parse_attribute(node.attribute, {
        #         "auto_pad": "NOTSET",
        #         "count_include_pad": 0,
        #         "dilations": [1, 1],
        #         "pads": [0, 0, 0, 0],
        #         "strides": [1, 1]})
        #     # No dilation in ONNX
        #     assert len(node.input) == 1
        #     assert len(node.output) == 1
        #     assert attrs["count_include_pad"] == 0 # To be consistent with operator.cc
        #     assert len(attrs["kernel_shape"]) == 2
        #     assert len(attrs["pads"]) == 4
        #     assert len(attrs["strides"]) == 2
        #     assert len(attrs["dilations"]) == 2
        #     assert attrs["pads"][0] == attrs["pads"][2]
        #     assert attrs["pads"][1] == attrs["pads"][3]
        #     g.avgpool(ts[node.input[0]], ts[node.output[0]],
        #             attrs["kernel_shape"][0], attrs["kernel_shape"][1],
        #             attrs["dilations"][0], attrs["dilations"][1],
        #             attrs["pads"][0], attrs["pads"][1],
        #             attrs["strides"][0], attrs["strides"][1])
    
        # elif node.op_type == 'Add':
        #     assert len(node.input) == 2
        #     assert len(node.output) == 1
        #     g.add(ts[node.input[0]], ts[node.input[1]], ts[node.output[0]])

        # elif node.op_type == 'Sub':
        #     assert len(node.input) == 2
        #     assert len(node.output) == 1
        #     g.sub(ts[node.input[0]], ts[node.input[1]], ts[node.output[0]])
    
        # elif node.op_type == 'Mul':
        #     assert len(node.input) == 2
        #     assert len(node.output) == 1
        #     g.mul(ts[node.input[0]], ts[node.input[1]], ts[node.output[0]])

        # elif node.op_type == 'Div':
        #     assert len(node.input) == 2
        #     assert len(node.output) == 1
        #     g.div(ts[node.input[0]], ts[node.input[1]], ts[node.output[0]])

        # elif node.op_type == 'Pow':
        #     assert len(node.input) == 2
        #     assert len(node.output) == 1
        #     g.pow(ts[node.input[0]], ts[node.input[1]], ts[node.output[0]])

        # elif node.op_type == 'Gather':
        #     attrs = _parse_attribute(node.attribute, {"axis": 0})
        #     assert len(node.input) == 2
        #     assert len(node.output) == 1
        #     g.gather(ts[node.input[0]], ts[node.input[1]], ts[node.output[0]], attrs["axis"])

        # elif node.op_type == 'Reshape':
        #     attrs = _parse_attribute(node.attribute, {})
        #     assert len(node.input) == 1
        #     assert len(node.output) == 1
        #     g.reshape(ts[node.input[0]], ts[node.output[0]], attrs["dims"])

        # elif node.op_type == 'Flatten':
        #     assert len(node.input) == 1
        #     assert len(node.output) == 1
        #     g.flatten(ts[node.input[0]], ts[node.output[0]])

        # elif node.op_type == 'Identity':
        #     assert len(node.input) == 1
        #     assert len(node.output) == 1
        #     g.identity(ts[node.input[0]], ts[node.output[0]])

        # elif node.op_type == 'Softmax':
        #     assert len(node.input) == 1
        #     assert len(node.output) == 1
        #     g.softmax(ts[node.input[0]], ts[node.output[0]])

        # elif node.op_type == 'Relu':
        #     assert len(node.input) == 1
        #     assert len(node.output) == 1
        #     g.relu(ts[node.input[0]], ts[node.output[0]])

        # elif node.op_type == 'Sigmoid':
        #     assert len(node.input) == 1
        #     assert len(node.output) == 1
        #     g.sigmoid(ts[node.input[0]], ts[node.output[0]])

        # elif node.op_type == 'Tanh':
        #     assert len(node.input) == 1
        #     assert len(node.output) == 1
        #     g.tanh(ts[node.input[0]], ts[node.output[0]])

        # elif node.op_type == 'Abs':
        #     assert len(node.input) == 1
        #     assert len(node.output) == 1
        #     g.abs(ts[node.input[0]], ts[node.output[0]])

    #     elif node.op_type == 'Cast':
    #         assert len(node.input) == 1
    #         assert len(node.output) == 1
    #         # Ignore for now (TODO)
    #         g.identity(ts[node.input[0]], ts[node.output[0]])

    #     elif node.op_type == 'Constant':
    #         attrs = _parse_attribute(node.attribute, {})
    #         assert len(node.output) == 1
    #         c = onnx.numpy_helper.to_array(attrs["value"])
    #         if c.ndim == 0:
    #             c = c[()]
    #         consts[node.output[0]] = c

    #     elif node.op_type == 'Gemm':
    #         attrs = _parse_attribute(node.attribute, {
    #             "alpha": 1.0,
    #             "beta": 1.0,
    #             "transA": 0,
    #             "transB": 0})
    #         assert len(node.input) == 2 or len(node.input) == 3
    #         assert len(node.output) == 1
    #         assert attrs["alpha"] == 1.0
    #         assert attrs["beta"] == 1.0 or len(node.input) == 2
    #         tmpI0 = g.tensor([1] + list(ds[node.input[0]]), "FLOAT")
    #         tmpI1 = g.tensor([1] + list(ds[node.input[1]]), "FLOAT")
    #         tmpO = g.tensor([1] + list(ds[node.output[0]]), "FLOAT")
    #         g.transpose(ts[node.input[0]], tmpI0, 0, Perm([PermItem(-1), PermItem(0), PermItem(1)]), 1)
    #         g.transpose(ts[node.input[1]], tmpI1, 0, Perm([PermItem(-1), PermItem(0), PermItem(1)]), 1)
    #         g.matmul(tmpI0, tmpI1, tmpO,
    #                 attrs["transA"], attrs["transB"],
    #                 None if len(node.input) == 2 else ts[node.input[2]])
    #         g.transpose(tmpO, ts[node.output[0]], -1, Perm([PermItem([0, 1]), PermItem(2)]), 0)

    #     elif node.op_type == 'GlobalAveragePool':
    #         assert len(node.input) == 1
    #         assert len(node.output) == 1
    #         dims = ds[node.input[0]]
    #         if len(dims) > 0:
    #             g.avgpool(ts[node.input[0]], ts[node.output[0]], dims[2], dims[3], 0, 0, 1, 1)
    #         else:
    #             g.avgpool(ts[node.input[0]], ts[node.output[0]])


    #     elif node.op_type == 'ReduceMean':
    #         attrs = _parse_attribute(node.attribute, {'keepdims': 1})
    #         assert len(node.input) == 1
    #         assert len(node.output) == 1
    #         assert len(attrs["axes"]) == 1
    #         axis = attrs["axes"][0]
    #         if axis < 0:
    #             axis = len(ds[node.input[0]]) - axis
    #         g.reduceMean(ts[node.input[0]], ts[node.output[0]], axis)

    #     elif node.op_type == 'Shape':
    #         # Ignore for now, and no need to output anything (TODO)
    #         pass

    #     elif node.op_type == 'Transpose':
    #         attrs = _parse_attribute(node.attribute, {})
    #         assert len(node.input) == 1
    #         assert len(node.output) == 1
    #         assert "perm" in attrs
    #         g.transpose(ts[node.input[0]], ts[node.output[0]], -1,
    #                 Perm([PermItem(x) for x in attrs["perm"]]), 0)

    #     elif node.op_type == 'Unsqueeze':
    #         assert len(node.input) == 2
    #         assert len(node.output) == 1
    #         g.reshape(ts[node.input[0]], ts[node.output[0]])
            
    #     elif node.op_type == "BatchNormalization":
    #         attrs = _parse_attribute(node.attribute, {})
    #         assert len(node.input) == 5
    #         assert len(node.output) == 1
    #         epsilon = attrs['epsilon'] if 'epsilon' in attrs else 1e-5
    #         momentum = attrs['momentum'] if 'momentum' in attrs else 0.9
    #         g.batchnorm(ts[node.input[0]], ts[node.input[1]],
    #                     ts[node.input[2]], ts[node.input[3]], 
    #                     ts[node.input[4]], ts[node.output[0]],
    #                     epsilon, momentum)

    #     elif node.op_type == "Resize":
    #         attrs = _parse_attribute(node.attribute, {})
    #         assert len(node.input) == 4
    #         assert len(node.output) == 1
    #         roi = ts[node.input[1]] if node.input[1] != '' else g.tensor(
    #             [], 'FLOAT')
    #         g.resize(ts[node.input[0]], roi, ts[node.output[0]])

        else:
            assert False, "Unsupported op: " + node.op_type
