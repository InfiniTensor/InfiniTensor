import os, onnx, unittest
from onnx import TensorProto
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info
from onnx.checker import check_model
from pyinfinitensor.onnx import from_onnx, parse_onnx, backend, runtime


class TestStringMethods(unittest.TestCase):
    def test_load(self):
        model_file = next(
            (name for name in os.listdir() if name.endswith(".onnx")), None
        )
        if model_file != None:
            print(
                "model: {file}({size:.2f} MiB)".format(
                    file=model_file, size=os.path.getsize(model_file) / 1024 / 1024
                )
            )
            parse_onnx(onnx.load(model_file))

    def test_tensor(self):
        x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3])
        graph = make_graph([], "tensor", [x], [x])
        model = make_model(graph)
        check_model(model)
        from_onnx(model)

    def test_matmul(self):
        x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3])
        a = make_tensor_value_info("a", TensorProto.FLOAT, [1, 3, 4])
        xa = make_tensor_value_info("b", TensorProto.FLOAT, [1, 2, 4])
        matmul = make_node("MatMul", ["x", "a"], ["xa"], name="matmul")
        graph = make_graph([matmul], "matmul", [x, a], [xa])
        model = make_model(graph)
        check_model(model)
        from_onnx(model)

    def test_add(self):
        a = make_tensor_value_info("a", TensorProto.FLOAT, [1, 3, 5, 7])
        b = make_tensor_value_info("b", TensorProto.FLOAT, [1, 3, 5, 7])
        c = make_tensor_value_info("c", TensorProto.FLOAT, [1, 3, 5, 7])
        add = make_node("Add", ["a", "b"], ["c"], name="add")
        graph = make_graph([add], "add", [a, b], [c])
        model = make_model(graph)
        check_model(model)
        from_onnx(model)

    def test_sub(self):
        a = make_tensor_value_info("a", TensorProto.FLOAT, [1, 3, 5, 7])
        b = make_tensor_value_info("b", TensorProto.FLOAT, [1, 3, 5, 7])
        c = make_tensor_value_info("c", TensorProto.FLOAT, [1, 3, 5, 7])
        sub = make_node("Sub", ["a", "b"], ["c"], name="sub")
        graph = make_graph([sub], "sub", [a, b], [c])
        model = make_model(graph)
        check_model(model)
        from_onnx(model)

    def test_mul(self):
        a = make_tensor_value_info("a", TensorProto.FLOAT, [1, 3, 5, 7])
        b = make_tensor_value_info("b", TensorProto.FLOAT, [1, 3, 5, 7])
        c = make_tensor_value_info("c", TensorProto.FLOAT, [1, 3, 5, 7])
        mul = make_node("Mul", ["a", "b"], ["c"], name="mul")
        graph = make_graph([mul], "mul", [a, b], [c])
        model = make_model(graph)
        check_model(model)
        from_onnx(model)

    def test_div(self):
        a = make_tensor_value_info("a", TensorProto.FLOAT, [1, 3, 5, 7])
        b = make_tensor_value_info("b", TensorProto.FLOAT, [1, 3, 5, 7])
        c = make_tensor_value_info("c", TensorProto.FLOAT, [1, 3, 5, 7])
        div = make_node("Div", ["a", "b"], ["c"], name="div")
        graph = make_graph([div], "div", [a, b], [c])
        model = make_model(graph)
        check_model(model)
        from_onnx(model)

    def test_pow(self):
        a = make_tensor_value_info("a", TensorProto.FLOAT, [1, 3, 5, 7])
        b = make_tensor_value_info("b", TensorProto.FLOAT, [1, 3, 5, 7])
        c = make_tensor_value_info("c", TensorProto.FLOAT, [1, 3, 5, 7])
        pow = make_node("Pow", ["a", "b"], ["c"], name="pow")
        graph = make_graph([pow], "pow", [a, b], [c])
        model = make_model(graph)
        check_model(model)
        from_onnx(model)

    def test_relu(self):
        x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 7])
        y = make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 5, 7])
        relu = make_node("Relu", ["x"], ["y"], name="relu")
        graph = make_graph([relu], "relu", [x], [y])
        model = make_model(graph)
        check_model(model)
        from_onnx(model)

    def test_sigmoid(self):
        x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 7])
        y = make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 5, 7])
        sigmoid = make_node("Sigmoid", ["x"], ["y"], name="sigmoid")
        graph = make_graph([sigmoid], "sigmoid", [x], [y])
        model = make_model(graph)
        check_model(model)
        from_onnx(model)

    def test_tanh(self):
        x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 7])
        y = make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 5, 7])
        tanh = make_node("Tanh", ["x"], ["y"], name="tanh")
        graph = make_graph([tanh], "tanh", [x], [y])
        model = make_model(graph)
        check_model(model)
        from_onnx(model)

    def test_softmax(self):
        x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 7])
        y = make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 5, 7])
        softmax = make_node("Softmax", ["x"], ["y"], name="softmax")
        graph = make_graph([softmax], "softmax", [x], [y])
        model = make_model(graph)
        check_model(model)
        from_onnx(model)

    def test_abs(self):
        x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 7])
        y = make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 5, 7])
        abs = make_node("Abs", ["x"], ["y"], name="abs")
        graph = make_graph([abs], "abs", [x], [y])
        model = make_model(graph)
        check_model(model)
        from_onnx(model)

    # see <https://onnx.ai/onnx/intro/python.html#a-simple-example-a-linear-regression>
    def test_linear(self):
        x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3])
        a = make_tensor_value_info("a", TensorProto.FLOAT, [1, 3, 4])
        b = make_tensor_value_info("b", TensorProto.FLOAT, [1, 2, 4])
        y = make_tensor_value_info("b", TensorProto.FLOAT, [1, 2, 4])
        matmul = make_node("MatMul", ["x", "a"], ["xa"], name="matmul")
        add = make_node("Add", ["xa", "b"], ["y"], name="add")
        graph = make_graph([matmul, add], "lr", [x, a, b], [y])
        model = make_model(graph)
        check_model(model)
        print(model)

        from_onnx(model)
        parse_onnx(model)

    def test_frontend(self):
        handler = backend.GraphHandlerObj(runtime)
        i = handler.tensor([1, 2, 3], 12)
        w = handler.tensor([1, 3, 4], 12)
        o = handler.tensor([1, 2, 4], 12)
        handler.matmul(i, w, o, False, False, None, backend.ActType.Relu)


if __name__ == "__main__":
    unittest.main()
