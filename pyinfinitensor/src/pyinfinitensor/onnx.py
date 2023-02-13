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
            tensors[node.output[0]] = handler.matmul(
                tensors[node.input[0]],
                tensors[node.input[1]],
                tensors.get(node.output[0], None),
                False,
                False,
                None,
                backend.ActType.Linear,
            )
        elif node.op_type == "Add":
            tensors[node.output[0]] = handler.add(
                tensors[node.input[0]],
                tensors[node.input[1]],
                tensors.get(node.output[0], None),
            )
        elif node.op_type == "Sub":
            tensors[node.output[0]] = handler.sub(
                tensors[node.input[0]],
                tensors[node.input[1]],
                tensors.get(node.output[0], None),
            )
        elif node.op_type == "Mul":
            tensors[node.output[0]] = handler.mul(
                tensors[node.input[0]],
                tensors[node.input[1]],
                tensors.get(node.output[0], None),
            )
        elif node.op_type == "Div":
            tensors[node.output[0]] = handler.div(
                tensors[node.input[0]],
                tensors[node.input[1]],
                tensors.get(node.output[0], None),
            )
        elif node.op_type == "Pow":
            tensors[node.output[0]] = handler.pow(
                tensors[node.input[0]],
                tensors[node.input[1]],
                tensors.get(node.output[0], None),
            )
        elif node.op_type == "Relu":
            tensors[node.output[0]] = handler.relu(
                tensors[node.input[0]],
                tensors.get(node.output[0], None),
            )
        elif node.op_type == "Sigmoid":
            tensors[node.output[0]] = handler.sigmoid(
                tensors[node.input[0]],
                tensors.get(node.output[0], None),
            )
        elif node.op_type == "Tanh":
            tensors[node.output[0]] = handler.tanh(
                tensors[node.input[0]],
                tensors.get(node.output[0], None),
            )
        elif node.op_type == "Softmax":
            tensors[node.output[0]] = handler.softmax(
                tensors[node.input[0]],
                tensors.get(node.output[0], None),
            )
        elif node.op_type == "Abs":
            tensors[node.output[0]] = handler.abs(
                tensors[node.input[0]],
                tensors.get(node.output[0], None),
            )
        elif node.op_type == "Identity":
            tensors[node.output[0]] = handler.identity(
                tensors[node.input[0]],
                tensors.get(node.output[0], None),
            )
        else:
            raise Exception('Unsupported operator "{}"'.format(node.op_type))


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
