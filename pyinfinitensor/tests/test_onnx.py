import os, onnx, unittest
from onnx import TensorProto
from onnx.helper import (
    make_model,
    make_node,
    make_tensor,
    make_graph,
    make_tensor_value_info,
)
from onnx.checker import check_model
from pyinfinitensor.onnx import from_onnx, parse_onnx, backend, runtime


def make_and_import_model(graph: onnx.GraphProto):
    model = make_model(graph)
    check_model(model)
    from_onnx(model)


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
        make_and_import_model(make_graph([], "tensor", [x], [x]))

    def test_matmul(self):
        x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3])
        a = make_tensor_value_info("a", TensorProto.FLOAT, [1, 3, 4])
        xa = make_tensor_value_info("b", TensorProto.FLOAT, [1, 2, 4])
        matmul = make_node("MatMul", ["x", "a"], ["xa"], name="matmul")
        make_and_import_model(make_graph([matmul], "matmul", [x, a], [xa]))

    def test_batch_norm(self):
        x = make_tensor_value_info("x", TensorProto.UINT32, [1, 3, 2, 2])
        scale = make_tensor_value_info("scale", TensorProto.FLOAT, [1, 3, 1, 1])
        b = make_tensor_value_info("b", TensorProto.FLOAT, [1, 3, 1, 1])
        mean = make_tensor_value_info("mean", TensorProto.FLOAT, [1, 3, 1, 1])
        var = make_tensor_value_info("var", TensorProto.FLOAT, [1, 3, 1, 1])
        y = make_tensor_value_info("y", TensorProto.UINT32, [1, 3, 2, 2])
        batch_norm = make_node(
            "BatchNormalization",
            ["x", "scale", "b", "mean", "var"],
            ["y"],
            name="batchNormalization",
        )
        make_and_import_model(
            make_graph([batch_norm], "batchNorm", [x, scale, b, mean, var], [y])
        )

    def test_max_pool(self):
        x = make_tensor_value_info("x", TensorProto.UINT32, [1, 64, 162, 162])
        y = make_tensor_value_info("y", TensorProto.UINT32, [1, 64, 80, 80])
        pool = make_node(
            "MaxPool",
            ["x"],
            ["y"],
            kernel_shape=[3, 3],
            dilations=[1, 1],
            pads=[0, 0],
            strides=[2, 2],
            name="maxPool",
        )
        make_and_import_model(make_graph([pool], "maxPool", [x], [y]))

    def test_avg_pool(self):
        x = make_tensor_value_info("x", TensorProto.UINT32, [1, 64, 162, 162])
        y = make_tensor_value_info("y", TensorProto.UINT32, [1, 64, 80, 80])
        pool = make_node(
            "AveragePool",
            ["x"],
            ["y"],
            kernel_shape=[3, 3],
            pads=[0, 0],
            strides=[2, 2],
            name="avgPool",
        )
        make_and_import_model(make_graph([pool], "avgPool", [x], [y]))

    def test_add(self):
        a = make_tensor_value_info("a", TensorProto.FLOAT, [1, 3, 5, 7])
        b = make_tensor_value_info("b", TensorProto.FLOAT, [1, 3, 5, 7])
        c = make_tensor_value_info("c", TensorProto.FLOAT, [1, 3, 5, 7])
        add = make_node("Add", ["a", "b"], ["c"], name="add")
        make_and_import_model(make_graph([add], "add", [a, b], [c]))

    def test_sub(self):
        a = make_tensor_value_info("a", TensorProto.FLOAT, [1, 3, 5, 7])
        b = make_tensor_value_info("b", TensorProto.FLOAT, [1, 3, 5, 7])
        c = make_tensor_value_info("c", TensorProto.FLOAT, [1, 3, 5, 7])
        sub = make_node("Sub", ["a", "b"], ["c"], name="sub")
        make_and_import_model(make_graph([sub], "sub", [a, b], [c]))

    def test_mul(self):
        a = make_tensor_value_info("a", TensorProto.FLOAT, [1, 3, 5, 7])
        b = make_tensor_value_info("b", TensorProto.FLOAT, [1, 3, 5, 7])
        c = make_tensor_value_info("c", TensorProto.FLOAT, [1, 3, 5, 7])
        mul = make_node("Mul", ["a", "b"], ["c"], name="mul")
        make_and_import_model(make_graph([mul], "mul", [a, b], [c]))

    def test_div(self):
        a = make_tensor_value_info("a", TensorProto.FLOAT, [1, 3, 5, 7])
        b = make_tensor_value_info("b", TensorProto.FLOAT, [1, 3, 5, 7])
        c = make_tensor_value_info("c", TensorProto.FLOAT, [1, 3, 5, 7])
        div = make_node("Div", ["a", "b"], ["c"], name="div")
        make_and_import_model(make_graph([div], "div", [a, b], [c]))

    def test_pow(self):
        a = make_tensor_value_info("a", TensorProto.FLOAT, [1, 3, 5, 7])
        b = make_tensor_value_info("b", TensorProto.FLOAT, [1, 3, 5, 7])
        c = make_tensor_value_info("c", TensorProto.FLOAT, [1, 3, 5, 7])
        pow = make_node("Pow", ["a", "b"], ["c"], name="pow")
        make_and_import_model(make_graph([pow], "pow", [a, b], [c]))

    def test_relu(self):
        x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 7])
        y = make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 5, 7])
        relu = make_node("Relu", ["x"], ["y"], name="relu")
        make_and_import_model(make_graph([relu], "relu", [x], [y]))

    def test_sigmoid(self):
        x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 7])
        y = make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 5, 7])
        sigmoid = make_node("Sigmoid", ["x"], ["y"], name="sigmoid")
        make_and_import_model(make_graph([sigmoid], "sigmoid", [x], [y]))

    def test_tanh(self):
        x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 7])
        y = make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 5, 7])
        tanh = make_node("Tanh", ["x"], ["y"], name="tanh")
        make_and_import_model(make_graph([tanh], "tanh", [x], [y]))

    def test_softmax(self):
        x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 7])
        y = make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 5, 7])
        softmax = make_node("Softmax", ["x"], ["y"], name="softmax")
        make_and_import_model(make_graph([softmax], "softmax", [x], [y]))

    def test_abs(self):
        x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 7])
        y = make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 5, 7])
        abs = make_node("Abs", ["x"], ["y"], name="abs")
        make_and_import_model(make_graph([abs], "abs", [x], [y]))

    def test_identity(self):
        x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 7])
        y = make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 5, 7])
        identity = make_node("Identity", ["x"], ["y"], name="identity")
        make_and_import_model(make_graph([identity], "identity", [x], [y]))

    def test_flatten(self):
        x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 7])
        y = make_tensor_value_info("y", TensorProto.FLOAT, [1 * 3 * 5 * 7])
        flatten = make_node("Flatten", ["x"], ["y"], name="flatten")
        make_and_import_model(make_graph([flatten], "flatten", [x], [y]))

    def test_reshape(self):
        data = make_tensor_value_info("data", TensorProto.FLOAT, [2, 3, 4, 5])
        # shape 对于后端来说并不是一个张量，然而转换中可能没有办法分辨
        # 不知道怎么把 ValueInfoProto 转换成 TensorProto
        shape = make_tensor_value_info("shape", TensorProto.INT64, [3])
        shape_data = make_tensor("shape", TensorProto.INT64, [3], [5, 3, 8])
        reshaped = make_tensor_value_info(
            "reshaped", TensorProto.FLOAT, shape_data.int64_data
        )
        reshape = make_node("Reshape", ["data", "shape"], ["reshaped"], name="reshape")
        # 可以构造一个 shape 只出现在 initializer 里而不出现在 input 里的图，
        # 但实际上的图中 initializer 里的必然会出现在 input 里，不知道为什么这样设计
        make_and_import_model(
            make_graph([reshape], "reshape", [data, shape], [reshaped], [shape_data])
        )

    def test_concat(self):
        input1 = make_tensor_value_info("input1", TensorProto.FLOAT, [1, 3, 2, 4])
        input2 = make_tensor_value_info("input2", TensorProto.FLOAT, [1, 3, 2, 5])
        output = make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 2, 9])
        concat = make_node(
            "Concat", ["input1", "input2"], ["output"], axis=3, name="concat"
        )
        make_and_import_model(
            make_graph([concat], "concat", [input1, input2], [output])
        )

    def test_gather(self):
        data = make_tensor_value_info("data", TensorProto.FLOAT, [1, 3, 4, 4])
        indices = make_tensor_value_info("indices", TensorProto.FLOAT, [2, 1, 2])
        output = make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 1, 2, 4, 4])
        gather = make_node(
            "Gather", ["data", "indices"], ["output"], axis=1, name="gather"
        )
        make_and_import_model(make_graph([gather], "gather", [data, indices], [output]))

    def test_reduce_mean(self):
        data = make_tensor_value_info("data", TensorProto.FLOAT, [2, 3, 3, 4])
        reduced = make_tensor_value_info("reduced", TensorProto.FLOAT, [1, 1, 1, 1])
        reduceMean = make_node(
            "ReduceMean", ["data"], ["reduced"], keepdims=1, name="reduceMean"
        )
        make_and_import_model(make_graph([reduceMean], "reduceMean", [data], [reduced]))

    def test_slice(self):
        data = make_tensor_value_info("data", TensorProto.UINT32, [10, 64, 162, 162])
        output = make_tensor_value_info("output", TensorProto.UINT32, [2, 1, 100, 96])
        starts = make_tensor_value_info("starts", TensorProto.INT64, [4])
        starts_data = make_tensor("starts", TensorProto.INT64, [4], [2, 10, 1, 5])
        ends = make_tensor_value_info("ends", TensorProto.INT64, [4])
        ends_data = make_tensor("ends", TensorProto.INT64, [4], [3, 10, 100, 100])
        slice = make_node(
            "Slice", ["data", "starts", "ends"], ["output"], name="gather"
        )
        make_and_import_model(
            make_graph(
                [slice],
                "slice",
                [data, starts, ends],
                [output],
                [starts_data, ends_data],
            )
        )

    # see <https://onnx.ai/onnx/intro/python.html#a-simple-example-a-linear-regression>
    def test_linear(self):
        x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3])
        a = make_tensor_value_info("a", TensorProto.FLOAT, [1, 3, 4])
        b = make_tensor_value_info("b", TensorProto.FLOAT, [1, 2, 4])
        y = make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 4])
        matmul = make_node("MatMul", ["x", "a"], ["xa"], name="matmul")
        add = make_node("Add", ["xa", "b"], ["y"], name="add")
        graph = make_graph([matmul, add], "lr", [x, a, b], [y])
        model = make_model(graph)
        check_model(model)
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
