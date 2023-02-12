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

    def test_import(self):
        i = make_tensor_value_info("i", TensorProto.FLOAT, [1, 2, 3])
        w = make_tensor_value_info("w", TensorProto.FLOAT, [1, 3, 4])
        o = make_tensor_value_info("o", TensorProto.FLOAT, [1, 2, 4])
        matmul = make_node("MatMul", ["i", "w"], ["o"], name="matmul")
        graph = make_graph([matmul], "mm", [i, w], [o])
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
