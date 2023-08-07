from onnx import TensorProto, TensorShapeProto, ModelProto, NodeProto, AttributeProto
from typing import Any
from onnx.numpy_helper import to_array


def _search_shape(model: ModelProto, name: str) -> list[int]:
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
            [int(d) for d in tensor.dims]
            for tensor in model.graph.initializer
            if tensor.name == name
        )
    )
    return ans


def _parse_attribute(node: NodeProto, attrs: dict[str, Any] = dict()) -> dict[str, Any]:
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


def _parse_all_attribute(node: NodeProto) -> dict[str, Any]:
    ans = dict()
    for attr in node.attribute:
        if attr.type == AttributeProto.INT:
            ans[attr.name] = attr.i
        elif attr.type == AttributeProto.INTS:
            ans[attr.name] = attr.ints
        elif attr.type == AttributeProto.FLOAT:
            ans[attr.name] = attr.f
        elif attr.type == AttributeProto.STRING:
            ans[attr.name] = attr.s
        elif attr.type == AttributeProto.TENSOR:
            ans[attr.name] = attr.t
        else:
            assert False, "Unsupported Attribute Type: {}".format(attr.type)
    return ans


def _parse_data(tensor: TensorProto) -> list[Any]:
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


def _take_shape_dim(shape: TensorShapeProto) -> list[int]:
    return [(d.dim_value if d.dim_value > 0 else 1) for d in shape.dim]
