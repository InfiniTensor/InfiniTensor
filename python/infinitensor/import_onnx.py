from pyinfinitensor import *

import functools

import numpy as np

import onnx
import onnx.checker
import onnx.numpy_helper
import onnx.shape_inference


def _add_value_info_for_constants(model: onnx.ModelProto):
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

    def add_const_value_infos_to_graph(graph: onnx.GraphProto):
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


def import_onnx(gf: GraphBuilder, net: str):
    ts, ds, ops, consts = dict(), dict(), dict(), dict()  # (key, value) = (name, class)
    model = onnx.load(net)

    # Tensor_input
    for input in model.graph.input:
        if input.name not in ts:
            dims = [d.dim_value for d in input.type.tensor_type.shape.dim]
            ts[input.name] = gf.tensor(dims, _onnx_datatype_tostring(
                input.type.tensor_type.elem_type))
            ds[input.name] = dims

    # Tensor_weight
    for weight in model.graph.initializer:
        if weight.name not in ts:
            ts[weight.name] = gf.tensor(
                weight.dims, _onnx_datatype_tostring(weight.data_type))
            ds[weight.name] = weight.dims

    # Tensor_inference
    _add_value_info_for_constants(model)
    infered_model = onnx.shape_inference.infer_shapes(model)
    for v in infered_model.graph.value_info:
        if v.name not in ts:
            dims = [d.dim_value for d in v.type.tensor_type.shape.dim]
            ts[v.name] = gf.tensor(dims, _onnx_datatype_tostring(
                v.type.tensor_type.elem_type))
            ds[v.name] = dims

    # Tensor_output
    for output in model.graph.output:
        if output.name not in ts:
            dims = [d.dim_value for d in output.type.tensor_type.shape.dim]
            ts[output.name] = gf.tensor(dims, _onnx_datatype_tostring(
                output.type.tensor_type.elem_type))
            ds[output.name] = dims

    # Op
    for node in model.graph.node:
        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv
        if node.op_type == 'Conv':
            attrs = _parse_attribute(node.attribute, {
                "auto_pad": "NOTSET",
                "dilations": [1, 1],
                "pads": [0, 0, 0, 0],
                "strides": [1, 1]})
            assert len(node.input) == 2  # bias is not implemented yet
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

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#MatMul
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
        #         g.transpose(ts[node.input[0]], tmpI0, 0, Perm([PermItem(-1), PermItem(0), PermItem(1)]), 1)
        #         g.transpose(ts[node.input[1]], tmpI1, 0, Perm([PermItem(-1), PermItem(0), PermItem(1)]), 1)
        #         g.matmul(tmpI0, tmpI1, tmpO, False, False, None)
        #         g.transpose(tmpO, ts[node.output[0]], -1, Perm([PermItem([0, 1]), PermItem(2)]), 0)

        #     else:
        #         assert len(dimO) >= 3
        #         batch = functools.reduce(lambda x, y: x * y, dimO[:-2])

        #         if len(dimA) == 3:
        #             tmpI0 = ts[node.input[0]]
        #         else:
        #             tmpI0 = g.tensor([batch, dimA[-2], dimA[-1]], "FLOAT")
        #             g.reshape(ts[node.input[0]], tmpI0)

        #         if len(dimB) == 3:
        #             tmpI1 = ts[node.input[1]]
        #         else:
        #             tmpI1 = g.tensor([batch, dimB[-2], dimB[-1]], "FLOAT")
        #             g.reshape(ts[node.input[1]], tmpI1)

        #         if len(dimO) == 3:
        #             tmpO = ts[node.output[0]]
        #             g.matmul(tmpI0, tmpI1, tmpO, False, False, None)
        #         else:
        #             tmpO = g.tensor([batch, dimO[-2], dimO[-1]], "FLOAT")
        #             g.matmul(tmpI0, tmpI1, tmpO, False, False, None)
        #             g.reshape(tmpO, ts[node.output[0]])

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvTranspose
        elif node.op_type == 'ConvTranspose':
            attrs = _parse_attribute(node.attribute, {
                "auto_pad": "NOTSET",
                "dilations": [1, 1],
                "pads": [0, 0, 0, 0],
                "strides": [1, 1],
                "group": 1})
            assert len(node.input) == 2 or len(node.input) == 3
            assert len(node.output) == 1
            assert attrs["auto_pad"] == "NOTSET"
            assert len(attrs["pads"]) == 4
            assert len(attrs["strides"]) == 2
            assert len(attrs["dilations"]) == 2
            assert attrs["pads"][0] == attrs["pads"][2]
            assert attrs["pads"][1] == attrs["pads"][3]
            oph, opw = 0, 0
            if "output_padding" in attrs:
                oph, opw = attrs["output_padding"][0], attrs["output_padding"][1]
                assert attrs["output_padding"][0] == attrs["output_padding"][1]
            gf.convTrans(ts[node.input[0]], ts[node.input[1]], ts[node.output[0]],
                         attrs["pads"][0], attrs["pads"][1],
                         attrs["strides"][0], attrs["strides"][1],
                         attrs["dilations"][0], attrs["dilations"][1],
                         oph, opw, group,
                         None if len(node.input) == 2 else ts[node.input[2]])

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Pad
        elif node.op_type == 'Pad':
            attrs = _parse_attribute(node.attribute, {'mode': b'constant'})
            assert attrs["mode"].decode("ascii") == "constant"
            assert len(attrs["pads"]) % 2 == 0
            if "constant_value" in attrs:
                assert attrs["constant_value"] == 0
            gf.pad(ts[node.input[0]], ts[node.output[0]], attrs["pads"],
                   attrs["axes"] if axes in attrs else None)

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Slice
        elif node.op_type == "Slice":
            assert 3 <= len(node.input) <= 5
            assert len(node.output) == 1
            gf.slice(ts[node.input[0]], ts[node.output[0]],
                     ts[node.input[1]], ts[node.input[2]],
                     ts[node.input[3]] if len(node.input) == 4 else None,
                     ts[node.input[4]] if len(node.input) == 5 else None)

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Concat
        elif node.op_type == 'Concat':
            attrs = _parse_attribute(node.attribute, {})
            assert len(node.output) == 1
            gf.concat([ts[item] for item in node.input],
                      ts[node.output[0]], attrs["axis"])

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Split
        elif node.op_type == "Split":
            attrs = _parse_attribute(node.attribute, {'axis': 0})
            assert len(node.input) == 1
            assert len(node.output) > 1
            dim = attrs['axis']
            num = attrs['num_outputs']
            gf.split(ts[node.input[0]], [ts[t] for t in node.output], dim, num)

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool
        elif node.op_type == 'MaxPool':
            attrs = _parse_attribute(node.attribute, {
                "auto_pad": "NOTSET",
                "dilations": [1, 1],
                "pads": [0, 0, 0, 0],
                "strides": [1, 1]})
            assert len(node.input) == 1
            assert len(node.output) == 1
            assert len(attrs["kernel_shape"]) == 2
            assert len(attrs["pads"]) == 4
            assert len(attrs["strides"]) == 2
            assert len(attrs["dilations"]) == 2
            assert attrs["pads"][0] == attrs["pads"][2]
            assert attrs["pads"][1] == attrs["pads"][3]
            gf.maxpool(ts[node.input[0]], ts[node.output[0]],
                       attrs["kernel_shape"][0], attrs["kernel_shape"][1],
                       attrs["dilations"][0], attrs["dilations"][1],
                       attrs["pads"][0], attrs["pads"][1],
                       attrs["strides"][0], attrs["strides"][1])

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#AveragePool
        # No dilation in ONNX
        elif node.op_type == 'AveragePool':
            attrs = _parse_attribute(node.attribute, {
                "auto_pad": "NOTSET",
                "count_include_pad": 0,
                "pads": [0, 0, 0, 0],
                "strides": [1, 1]})
            assert len(node.input) == 1
            assert len(node.output) == 1
            # To be consistent with operator.cc
            assert attrs["count_include_pad"] == 0
            assert len(attrs["kernel_shape"]) == 2
            assert len(attrs["pads"]) == 4
            assert len(attrs["strides"]) == 2
            assert attrs["pads"][0] == attrs["pads"][2]
            assert attrs["pads"][1] == attrs["pads"][3]
            dh, dw = 1, 1
            gf.avgpool(ts[node.input[0]], ts[node.output[0]],
                       attrs["kernel_shape"][0], attrs["kernel_shape"][1],
                       dw, dh,
                       attrs["pads"][0], attrs["pads"][1],
                       attrs["strides"][0], attrs["strides"][1])

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Add
        elif node.op_type == 'Add':
            assert len(node.input) == 2
            assert len(node.output) == 1
            assert ds[node.input[0]] == ds[node.output[0]]
            if ds[node.input[0]] == ds[node.input[1]]:
                gf.add(ts[node.input[0]], ts[node.input[1]],
                       ts[node.output[0]])
            elif len(ds[node.input[1]]) == 0:
                tmp = gf.tensor(ds[node.input[0]], "FLOAT")
                gf.add(ts[node.input[0]], tmp, ts[node.output[0]])
            elif ds[node.input[1]][-1] == 1 and ds[node.input[0]][:-1] == ds[node.input[1]][:-1]:
                tmp = gf.tensor(ds[node.output[0]], "FLOAT")
                gf.extend(ts[node.input[1]], tmp,
                          len(ds[node.input[1]]) - 1, ds[node.input[0]][-1] - 1)
                gf.add(ts[node.input[0]], tmp, ts[node.output[0]])
            elif len(ds[node.input[1]]) == 1 and ds[node.input[0]][-1] == ds[node.input[1]][0]:
                tmp = gf.tensor(ds[node.output[0]], "FLOAT")
                # gf.extend(ts[node.input[1]], tmp,
                #           len(ds[node.input[1]]) - 1, ds[node.input[0]][-1] - 1)
                gf.add(ts[node.input[0]], tmp, ts[node.output[0]])
            else:
                assert False

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sub
        elif node.op_type == 'Sub':
            assert len(node.input) == 2
            assert len(node.output) == 1
            assert ds[node.input[0]] == ds[node.output[0]]
            if ds[node.input[0]] == ds[node.input[1]]:
                gf.sub(ts[node.input[0]], ts[node.input[1]],
                       ts[node.output[0]])
            elif len(ds[node.input[1]]) == 0:
                tmp = gf.tensor(ds[node.input[0]], "FLOAT")
                gf.sub(ts[node.input[0]], tmp, ts[node.output[0]])
            elif ds[node.input[1]][-1] == 1 and ds[node.input[0]][:-1] == ds[node.input[1]][:-1]:
                tmp = gf.tensor(ds[node.output[0]], "FLOAT")
                gf.extend(ts[node.input[1]], tmp,
                          len(ds[node.input[1]]) - 1, ds[node.input[0]][-1] - 1)
                gf.sub(ts[node.input[0]], tmp, ts[node.output[0]])
            else:
                assert False

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mul
        elif node.op_type == 'Mul':
            assert len(node.input) == 2
            assert len(node.output) == 1
            assert ds[node.input[0]] == ds[node.output[0]]
            if ds[node.input[0]] == ds[node.input[1]]:
                gf.mul(ts[node.input[0]], ts[node.input[1]],
                       ts[node.output[0]])
            elif len(ds[node.input[1]]) == 0:
                tmp = gf.tensor(ds[node.input[0]], "FLOAT")
                gf.mul(ts[node.input[0]], tmp, ts[node.output[0]])
            elif ds[node.input[1]][-1] == 1 and ds[node.input[0]][:-1] == ds[node.input[1]][:-1]:
                tmp = gf.tensor(ds[node.output[0]], "FLOAT")
                gf.extend(ts[node.input[1]], tmp,
                          len(ds[node.input[1]]) - 1, ds[node.input[0]][-1] - 1)
                gf.mul(ts[node.input[0]], tmp, ts[node.output[0]])
            elif len(ds[node.input[1]]) == 1 and ds[node.input[0]][-1] == ds[node.input[1]][0]:
                tmp = gf.tensor(ds[node.output[0]], "FLOAT")
                # gf.extend(ts[node.input[1]], tmp,
                #           len(ds[node.input[1]]) - 1, ds[node.input[0]][-1] - 1)
                gf.mul(ts[node.input[0]], tmp, ts[node.output[0]])
            else:
                print(ds[node.input[0]], ds[node.input[1]])
                assert False

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Div
        elif node.op_type == 'Div':
            assert len(node.input) == 2
            assert len(node.output) == 1
            assert ds[node.input[0]] == ds[node.output[0]]
            if ds[node.input[0]] == ds[node.input[1]]:
                gf.div(ts[node.input[0]], ts[node.input[1]],
                       ts[node.output[0]])
            elif len(ds[node.input[1]]) == 0:
                tmp = gf.tensor(ds[node.input[0]], "FLOAT")
                gf.div(ts[node.input[0]], tmp, ts[node.output[0]])
            elif ds[node.input[1]][-1] == 1 and ds[node.input[0]][:-1] == ds[node.input[1]][:-1]:
                tmp = gf.tensor(ds[node.output[0]], "FLOAT")
                gf.extend(ts[node.input[1]], tmp,
                          len(ds[node.input[1]]) - 1, ds[node.input[0]][-1] - 1)
                gf.div(ts[node.input[0]], tmp, ts[node.output[0]])
            else:
                assert False

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Pow
        elif node.op_type == 'Pow':
            assert len(node.input) == 2
            assert len(node.output) == 1
            assert ds[node.input[0]] == ds[node.output[0]]
            if ds[node.input[0]] == ds[node.input[1]]:
                gf.pow(ts[node.input[0]], ts[node.input[1]],
                       ts[node.output[0]])
            elif len(ds[node.input[1]]) == 0:
                tmp = gf.tensor(ds[node.input[0]], "FLOAT")
                gf.pow(ts[node.input[0]], tmp, ts[node.output[0]])
            elif ds[node.input[1]][-1] == 1 and ds[node.input[0]][:-1] == ds[node.input[1]][:-1]:
                tmp = gf.tensor(ds[node.output[0]], "FLOAT")
                gf.extend(ts[node.input[1]], tmp,
                          len(ds[node.input[1]]) - 1, ds[node.input[0]][-1] - 1)
                gf.pow(ts[node.input[0]], tmp, ts[node.output[0]])
            else:
                assert False

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather
        elif node.op_type == 'Gather':
            attrs = _parse_attribute(node.attribute, {"axis": 0})
            assert len(node.input) == 2
            assert len(node.output) == 1
            gf.gather(ts[node.input[0]], ts[node.input[1]],
                      ts[node.output[0]], attrs["axis"])

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Reshape
        elif node.op_type == 'Reshape':
            attrs = _parse_attribute(node.attribute, {"allowzero": 0})
            assert len(node.input) == 2
            assert len(node.output) == 1
            gf.reshape(ts[node.input[0]],
                       ts[node.output[0]], ts[node.input[1]])

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Flatten
        # Output is 2D in ONNX
        elif node.op_type == 'Flatten':
            assert len(node.input) == 1
            assert len(node.output) == 1
            gf.flatten(ts[node.input[0]], ts[node.output[0]])

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Identity
        elif node.op_type == 'Identity':
            assert len(node.input) == 1
            assert len(node.output) == 1
            gf.identity(ts[node.input[0]], ts[node.output[0]])

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softmax
        elif node.op_type == 'Softmax':
            attrs = _parse_attribute(node.attribute, {"axis": -1})
            assert len(node.input) == 1
            assert len(node.output) == 1
            gf.softmax(ts[node.input[0]], ts[node.output[0]])

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Relu
        elif node.op_type == 'Relu':
            assert len(node.input) == 1
            assert len(node.output) == 1
            gf.relu(ts[node.input[0]], ts[node.output[0]])

        # TODO
        elif node.op_type == 'Sqrt':
            assert len(node.input) == 1
            assert len(node.output) == 1
            gf.relu(ts[node.input[0]], ts[node.output[0]])

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sigmoid
        elif node.op_type == 'Sigmoid':
            assert len(node.input) == 1
            assert len(node.output) == 1
            gf.sigmoid(ts[node.input[0]], ts[node.output[0]])

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tanh
        elif node.op_type == 'Tanh':
            assert len(node.input) == 1
            assert len(node.output) == 1
            gf.tanh(ts[node.input[0]], ts[node.output[0]])

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Abs
        elif node.op_type == 'Abs':
            assert len(node.input) == 1
            assert len(node.output) == 1
            gf.abs(ts[node.input[0]], ts[node.output[0]])

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cast
        # Ignore for now (TODO)
        elif node.op_type == 'Cast':
            assert len(node.input) == 1
            assert len(node.output) == 1
            gf.identity(ts[node.input[0]], ts[node.output[0]])

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Constant
        elif node.op_type == 'Constant':
            attrs = _parse_attribute(node.attribute, {})
            assert len(node.output) == 1
            c = onnx.numpy_helper.to_array(attrs["value"])
            if c.ndim == 0:
                c = c[()]
            consts[node.output[0]] = c

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm
        # elif node.op_type == 'Gemm':
        #     attrs = _parse_attribute(node.attribute, {
        #         "alpha": 1.0,
        #         "beta": 1.0,
        #         "transA": 0,
        #         "transB": 0})
        #     assert len(node.input) == 2 or len(node.input) == 3
        #     assert len(node.output) == 1
        #     assert attrs["alpha"] == 1.0
        #     assert attrs["beta"] == 1.0 or len(node.input) == 2
        #     tmpI0 = g.tensor([1] + list(ds[node.input[0]]), "FLOAT")
        #     tmpI1 = g.tensor([1] + list(ds[node.input[1]]), "FLOAT")
        #     tmpO = g.tensor([1] + list(ds[node.output[0]]), "FLOAT")
        #     g.transpose(ts[node.input[0]], tmpI0, 0, Perm([PermItem(-1), PermItem(0), PermItem(1)]), 1)
        #     g.transpose(ts[node.input[1]], tmpI1, 0, Perm([PermItem(-1), PermItem(0), PermItem(1)]), 1)
        #     g.matmul(tmpI0, tmpI1, tmpO,
        #             attrs["transA"], attrs["transB"],
        #             None if len(node.input) == 2 else ts[node.input[2]])
        #     g.transpose(tmpO, ts[node.output[0]], -1, Perm([PermItem([0, 1]), PermItem(2)]), 0)

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalAveragePool
        # elif node.op_type == 'GlobalAveragePool':
        #     assert len(node.input) == 1
        #     assert len(node.output) == 1
        #     dims = ds[node.input[0]]
        #     if len(dims) > 0:
        #         g.avgpool(ts[node.input[0]], ts[node.output[0]], dims[2], dims[3], 0, 0, 1, 1)
        #     else:
        #         g.avgpool(ts[node.input[0]], ts[node.output[0]])

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMean
        elif node.op_type == 'ReduceMean':
            attrs = _parse_attribute(node.attribute, {'keepdims': 1})
            assert len(node.input) == 1
            assert len(node.output) == 1
            assert len(attrs["axes"]) == 1
            axis = attrs["axes"][0]
            print(axis, len(ds[node.input[0]]))
            if axis < 0:
                axis = len(ds[node.input[0]]) + axis
            gf.reduceMean(ts[node.input[0]], ts[node.output[0]], axis)

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Shape
        # Ignore for now, and no need to output anything (TODO)
        # elif node.op_type == 'Shape':
        #     pass

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Transpose
        # elif node.op_type == 'Transpose':
        #     attrs = _parse_attribute(node.attribute, {})
        #     assert len(node.input) == 1
        #     assert len(node.output) == 1
        #     assert "perm" in attrs
        #     gf.transpose(ts[node.input[0]], ts[node.output[0]], -1,
        #             Perm([PermItem(x) for x in attrs["perm"]]), 0)

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Unsqueeze
        elif node.op_type == 'Unsqueeze':
            assert len(node.input) == 2
            assert len(node.output) == 1
            gf.reshape(ts[node.input[0]],
                       ts[node.output[0]], ts[node.input[1]])

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#BatchNormalization
        # elif node.op_type == "BatchNormalization":
        #     attrs = _parse_attribute(node.attribute, {})
        #     assert len(node.input) == 5
        #     assert len(node.output) == 1
        #     epsilon = attrs['epsilon'] if 'epsilon' in attrs else 1e-5
        #     momentum = attrs['momentum'] if 'momentum' in attrs else 0.9
        #     g.batchnorm(ts[node.input[0]], ts[node.input[1]],
        #                 ts[node.input[2]], ts[node.input[3]],
        #                 ts[node.input[4]], ts[node.output[0]],
        #                 epsilon, momentum)

        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Resize
        # elif node.op_type == "Resize":
        #     attrs = _parse_attribute(node.attribute, {})
        #     assert len(node.input) == 4
        #     assert len(node.output) == 1
        #     roi = ts[node.input[1]] if node.input[1] != '' else g.tensor(
        #         [], 'FLOAT')
        #     g.resize(ts[node.input[0]], roi, ts[node.output[0]])

        else:
            assert False, "Unsupported op: " + node.op_type
