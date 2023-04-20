import os, onnx, unittest
from onnx import TensorProto
from onnx.helper import (
    make_model,
    make_node,
    make_tensor,
    make_graph,
    make_tensor_value_info,
)
from onnx.checker import check_model, check_graph
from onnx.shape_inference import infer_shapes
from pyinfinitensor.onnx import from_onnx, OnnxStub, backend


def make_and_import_model(graph: onnx.GraphProto):
    check_graph(graph)
    model = make_model(graph)
    check_model(model)
    from_onnx(model, backend.cpu_runtime())


class TestStringMethods(unittest.TestCase):
    # def test_run(self):
    #    model_file = next(
    #        (name for name in os.listdir() if name.endswith(".onnx")), None
    #    )
    #    if model_file != None:
    #        print(
    #            "model: {file}({size:.2f} MiB)".format(
    #                file=model_file, size=os.path.getsize(model_file) / 1024 / 1024
    #            )
    #        )
    #        run_onnx(onnx.load(model_file), runtime)

    def test_load(self):
        for model_file in os.listdir():
            if model_file.endswith(".onnx"):
                print(
                    "model: {file}({size:.2f} MiB)".format(
                        file=model_file, size=os.path.getsize(model_file) / 1024 / 1024
                    )
                )
                model = OnnxStub.from_onnx(
                    onnx.load(model_file), backend.cpu_runtime()
                ).to_onnx("new")
                model = infer_shapes(model)

    def test_tensor(self):
        x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3])
        make_and_import_model(make_graph([], "tensor", [x], [x]))

    def test_conv(self):
        i = make_tensor_value_info("i", TensorProto.FLOAT, [1, 3, 4, 4])
        w = make_tensor_value_info("w", TensorProto.FLOAT, [2, 3, 3, 3])
        o = make_tensor_value_info("o", TensorProto.FLOAT, [1, 2, 2, 2])
        conv = make_node(
            "Conv",
            ["i", "w"],
            ["o"],
            "conv",
            pads=[1, 1, 1, 1],
            strides=[2, 1],
            dilations=[1, 2],
        )
        make_and_import_model(make_graph([conv], "conv", [i, w], [o]))

    def test_matmul(self):
        x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3])
        a = make_tensor_value_info("a", TensorProto.FLOAT, [1, 3, 4])
        xa = make_tensor_value_info("xa", TensorProto.FLOAT, [1, 2, 4])
        matmul = make_node("MatMul", ["x", "a"], ["xa"], name="matmul")
        make_and_import_model(make_graph([matmul], "matmul", [x, a], [xa]))

    def test_gemm(self):
        a = make_tensor_value_info("a", TensorProto.FLOAT, [1, 2, 3])
        b = make_tensor_value_info("b", TensorProto.FLOAT, [1, 4, 3])
        c = make_tensor_value_info("c", TensorProto.FLOAT, [1, 2, 4])
        y = make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 4])
        gemm = make_node("Gemm", ["a", "b", "c"], ["y"], transB=1, name="gemm")
        make_and_import_model(make_graph([gemm], "gemm", [a, b, c], [y]))

    def test_batch_norm(self):
        x = make_tensor_value_info("x", TensorProto.UINT32, [1, 3, 2, 2])
        scale = make_tensor_value_info("scale", TensorProto.FLOAT, [3])
        b = make_tensor_value_info("b", TensorProto.FLOAT, [3])
        mean = make_tensor_value_info("mean", TensorProto.FLOAT, [3])
        var = make_tensor_value_info("var", TensorProto.FLOAT, [3])
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
            pads=[0, 0, 0, 0],
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
            pads=[0, 0, 0, 0],
            strides=[2, 2],
            name="avgPool",
        )
        make_and_import_model(make_graph([pool], "avgPool", [x], [y]))

    def test_global_avg_pool(self):
        x = make_tensor_value_info("x", TensorProto.UINT32, [30, 30, 30, 30])
        y = make_tensor_value_info("y", TensorProto.UINT32, [30, 30, 1, 1])
        pool = make_node(
            "GlobalAveragePool",
            ["x"],
            ["y"],
            name="globalAvgPool",
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
        softmax = make_node("Softmax", ["x"], ["y"], axis=2, name="softmax")
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
        y = make_tensor_value_info("y", TensorProto.FLOAT, [1 * 3, 5 * 7])
        flatten = make_node("Flatten", ["x"], ["y"], axis=2, name="flatten")
        # make_and_import_model(
        make_graph([flatten], "flatten", [x], [y])
        # )

    def test_reshape(self):
        data = make_tensor_value_info("data", TensorProto.FLOAT, [2, 3, 4, 5])
        shape = make_tensor_value_info("shape", TensorProto.INT64, [3])
        shape_data = make_tensor("shape", TensorProto.INT64, [3], [5, 3, 8])
        reshaped = make_tensor_value_info(
            "reshaped", TensorProto.FLOAT, shape_data.int64_data
        )
        reshape = make_node("Reshape", ["data", "shape"], ["reshaped"], name="reshape")
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
        output = make_tensor_value_info("output", TensorProto.UINT32, [1, 1, 99, 95])
        starts = make_tensor("starts", TensorProto.INT64, [4], [2, 9, 1, 5])
        ends = make_tensor("ends", TensorProto.INT64, [4], [3, 10, 100, 100])
        slice = make_node("Slice", ["data", "starts", "ends"], ["output"], name="slice")
        make_and_import_model(
            make_graph(
                [slice],
                "slice",
                [data],
                [output],
                [starts, ends],
            )
        )

    def test_pad(self):
        data = make_tensor_value_info("data", TensorProto.UINT32, [1, 64, 162, 162])
        output = make_tensor_value_info("output", TensorProto.UINT32, [3, 84, 164, 172])
        pads = make_tensor_value_info("pads", TensorProto.INT64, [8])
        pads_data = make_tensor(
            "pads", TensorProto.INT64, [8], [2, 10, 1, 5, 0, 10, 1, 5]
        )
        pad = make_node("Pad", ["data", "pads"], ["output"], name="pad")
        make_and_import_model(
            make_graph(
                [pad],
                "pad",
                [data, pads],
                [output],
                [pads_data],
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
        from_onnx(model, backend.cpu_runtime())

    def test_frontend(self):
        handler = backend.GraphHandler(backend.cpu_runtime())
        a = handler.tensor([1, 2, 3], 12, backend.TensorType.Input)
        b = handler.tensor([1, 2, 3], 12, backend.TensorType.Input)
        c = handler.tensor([1, 2, 3], 12, backend.TensorType.Input)
        d = handler.tensor([1, 2, 3], 12, backend.TensorType.Input)
        e = handler.tensor([1, 2, 3], 12, backend.TensorType.Input)

        x = handler.add(
            handler.add(handler.add(handler.add(a, b, None), c, None), d, None), e, None
        )
        y = handler.tensor([3, 2, 1], 12, backend.TensorType.Other)
        handler.reshape(x, y, [3, 2, 1])


if __name__ == "__main__":
    unittest.main()
