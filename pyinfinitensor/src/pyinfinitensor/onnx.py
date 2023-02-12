import onnx
import backend

runtime = backend.cpu_runtime()


def from_onnx(model: onnx.ModelProto):
    handler = backend.GraphHandlerObj(runtime)

    tensors = dict()

    for input in model.graph.input:
        dims = [d.dim_value for d in input.type.tensor_type.shape.dim]
        tensors[input.name] = handler.tensor(dims, input.type.tensor_type.elem_type)

    for output in model.graph.output:
        dims = [d.dim_value for d in output.type.tensor_type.shape.dim]
        tensors[output.name] = handler.tensor(dims, output.type.tensor_type.elem_type)

    for node in model.graph.node:
        if node.op_type == "MatMul":
            handler.matmul(
                tensors[node.input[0]],
                tensors[node.input[1]],
                tensors[node.output[0]],
                False,
                False,
                None,
                backend.ActType.Linear,
            )


def parse_onnx(model: onnx.ModelProto):
    print()

    for field in [
        "doc_string",
        "domain",
        "functions",
        "metadata_props",
        "model_version",
        "producer_name",
        "producer_version",
        "training_info",
    ]:
        print("{}: {}".format(field, getattr(model, field)))

    print("ir_version:", model.ir_version)
    for opset in model.opset_import:
        print("opset domain={} version={}".format(opset.domain, opset.version))

    print("layout:")
    for node in model.graph.node:
        print(
            '   {o} <- {op}"{name}"{a} <- {i}'.format(
                name=node.name,
                op=node.op_type,
                i=node.input,
                o=node.output,
                a=[a.name for a in node.attribute],
            )
        )

    print("weight:")
    for node in model.graph.initializer:
        print("   {}".format(node.name))
