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
from pyinfinitensor.onnx import from_onnx, OnnxStub, backend, _parse_data_fp16
import numpy as np


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
                model_file = "/home/featurize/work/my_infiniTensor/InfiniTensor/resnet18-v2-7.onnx"
                model = OnnxStub(onnx.load(model_file), backend.cpu_runtime()).to_onnx(
                    "new"
                )
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

    def test_conv_fp16(self):
        i = make_tensor_value_info("i", TensorProto.FLOAT16, [1, 3, 4, 4])
        w = make_tensor_value_info("w", TensorProto.FLOAT16, [2, 3, 3, 3])
        o = make_tensor_value_info("o", TensorProto.FLOAT16, [1, 2, 2, 2])
        conv = make_node(
            "Conv",
            ["i", "w"],
            ["o"],
            "conv",
            pads=[1, 1, 1, 1],
            strides=[2, 1],
            dilations=[1, 2],
        )
        make_and_import_model(make_graph([conv], "conv_fp16", [i, w], [o]))

    def test_conv_bfp16(self):
        i = make_tensor_value_info("i", TensorProto.BFLOAT16, [1, 3, 4, 4])
        w = make_tensor_value_info("w", TensorProto.BFLOAT16, [2, 3, 3, 3])
        o = make_tensor_value_info("o", TensorProto.BFLOAT16, [1, 2, 2, 2])
        conv = make_node(
            "Conv",
            ["i", "w"],
            ["o"],
            "conv",
            pads=[1, 1, 1, 1],
            strides=[2, 1],
            dilations=[1, 2],
        )
        make_and_import_model(make_graph([conv], "conv_bfp16", [i, w], [o]))

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
            make_graph([batch_norm], "batchNormalzation", [x, scale, b, mean, var], [y])
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

    def test_leaky_relu(self):
        # Define input and output tensor information
        x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])
        y = make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4, 4])

        # Define the LeakyRelu node
        leaky_relu = make_node(
            "LeakyRelu",
            ["x"],
            ["y"],
            "leaky_relu",
            alpha=0.01  # LeakyReLU alpha value
        )

        # Create the graph and model
        graph = make_graph([leaky_relu], "leaky_relu", [x], [y])
        make_and_import_model(graph)

    """Gelu operator is not supported by onnx 14.1 currently."""

    def test_gelu(self):
        pass
        # x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 7])
        # y = make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 5, 7])
        # gelu = make_node("Gelu", ["x"], ["y"], name="gelu")
        # make_and_import_model(make_graph([gelu], "gelu", [x], [y]))

    def test_erf(self):
        x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 7])
        y = make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 5, 7])
        erf = make_node("Erf", ["x"], ["y"], name="erf")
        make_and_import_model(make_graph([erf], "erf", [x], [y]))

    def test_sqrt(self):
        x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 7])
        y = make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 5, 7])
        sqrt = make_node("Sqrt", ["x"], ["y"], name="sqrt")
        make_and_import_model(make_graph([sqrt], "sqrt", [x], [y]))

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

    def test_hard_sigmoid(self):
        x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 7])
        y = make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 5, 7])
        hardSigmoid = make_node("HardSigmoid", ["x"], ["y"], name="hardSigmoid")
        make_and_import_model(make_graph([hardSigmoid], "hardSigmoid", [x], [y]))

    def test_hard_swish(self):
        x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 7])
        y = make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 5, 7])
        hardSwish = make_node("HardSwish", ["x"], ["y"], name="hardSwish")
        make_and_import_model(make_graph([hardSwish], "hardSwish", [x], [y]))

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

    def test_neg(self):
        x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 7])
        y = make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 5, 7])
        neg = make_node("Neg", ["x"], ["y"], name="neg")
        make_and_import_model(make_graph([neg], "neg", [x], [y]))

    def test_identity(self):
        x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 7])
        y = make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 5, 7])
        identity = make_node("Identity", ["x"], ["y"], name="identity")
        make_and_import_model(make_graph([identity], "identity", [x], [y]))

    def test_flatten(self):
        x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 7])
        y = make_tensor_value_info("y", TensorProto.FLOAT, [1 * 3, 5 * 7])
        flatten = make_node("Flatten", ["x"], ["y"], axis=2, name="flatten")
        make_and_import_model(make_graph([flatten], "flatten", [x], [y]))

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

    def test_resize(self):
        x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 128, 40, 40])
        roi = make_tensor("roi", TensorProto.FLOAT, [0], [])
        scales = make_tensor("scales", TensorProto.FLOAT, [4], [1, 1, 2, 2])
        y = make_tensor_value_info("y", TensorProto.FLOAT, [1, 128, 80, 80])
        reshape = make_node("Resize", ["x", "roi", "scales"], ["y"], name="resize")
        make_and_import_model(make_graph([reshape], "resize", [x], [y], [roi, scales]))

    def test_squeeze(self):
        input = make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 1, 5])
        axes = make_tensor_value_info("axes", TensorProto.INT64, [2])
        axes_data = make_tensor("axes", TensorProto.INT64, [2], [0, 2])
        output = make_tensor_value_info("output", TensorProto.FLOAT, [3, 5])
        squeeze = make_node("Squeeze", ["input", "axes"], ["output"], name="squeeze")
        make_and_import_model(
            make_graph([squeeze], "squeeze", [input, axes], [output], [axes_data])
        )

    def test_unsqueeze(self):
        input = make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 4, 5])
        axes = make_tensor_value_info("axes", TensorProto.INT64, [2])
        axes_data = make_tensor("axes", TensorProto.INT64, [2], [0, 2])
        output = make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 1, 3, 4, 5])
        unsqueeze = make_node(
            "Unsqueeze", ["input", "axes"], ["output"], name="unsqueeze"
        )
        make_and_import_model(
            make_graph([unsqueeze], "unsqueeze", [input, axes], [output], [axes_data])
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
        indices = make_tensor_value_info("indices", TensorProto.INT64, [2, 1, 2])
        output = make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 1, 2, 4, 4])
        gather = make_node(
            "Gather", ["data", "indices"], ["output"], axis=1, name="gather"
        )
        make_and_import_model(make_graph([gather], "gather", [data, indices], [output]))

    def test_gather_elements(self):
        data = make_tensor_value_info("data", TensorProto.FLOAT, [2, 3, 2])
        indices = make_tensor_value_info("indices", TensorProto.INT64, [2, 1, 2])
        output = make_tensor_value_info("output", TensorProto.FLOAT, [2, 1, 2])
        gatherElements = make_node(
            "GatherElements",
            ["data", "indices"],
            ["output"],
            axis=1,
            name="gatherElements",
        )
        make_and_import_model(
            make_graph([gatherElements], "gatherElements", [data, indices], [output])
        )

    def test_reduce_mean(self):
        data = make_tensor_value_info("data", TensorProto.FLOAT, [2, 3, 3, 4])
        reduced = make_tensor_value_info("reduced", TensorProto.FLOAT, [1, 1, 1, 1])
        reduceMean = make_node(
            "ReduceMean", ["data"], ["reduced"], keepdims=1, name="reduceMean"
        )
        make_and_import_model(make_graph([reduceMean], "reduceMean", [data], [reduced]))

    def test_reduce_sum(self):
        data = make_tensor_value_info("data", TensorProto.FLOAT, [2, 3, 3, 4])
        reduced = make_tensor_value_info("reduced", TensorProto.FLOAT, [1, 1, 1, 1])
        reduceSum = make_node(
            "ReduceSum", ["data"], ["reduced"], keepdims=1, name="reduceSum"
        )
        make_and_import_model(make_graph([reduceSum], "reduceSum", [data], [reduced]))

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

    def test_allReduceSum(self):
        input = make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 2, 4])
        output = make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 2, 4])
        allReduceSum = make_node(
            "AllReduceSum", ["input"], ["output"], name="allReduceSum"
        )
        graph = make_graph([allReduceSum], "allReduceSum", [input], [output])
        model = make_model(graph)
        from_onnx(model, backend.cpu_runtime())

    def test_allReduceProd(self):
        input = make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 2, 4])
        output = make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 2, 4])
        allReduceProd = make_node(
            "AllReduceProd", ["input"], ["output"], name="allReduceProd"
        )
        graph = make_graph([allReduceProd], "allReduceProd", [input], [output])
        model = make_model(graph)
        from_onnx(model, backend.cpu_runtime())

    def test_allReduceMin(self):
        input = make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 2, 4])
        output = make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 2, 4])
        allReduceMin = make_node(
            "AllReduceMin", ["input"], ["output"], name="allReduceMin"
        )
        graph = make_graph([allReduceMin], "allReduceMin", [input], [output])
        model = make_model(graph)
        from_onnx(model, backend.cpu_runtime())

    def test_allReduceMax(self):
        input = make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 2, 4])
        output = make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 2, 4])
        allReduceMax = make_node(
            "AllReduceMax", ["input"], ["output"], name="allReduceMax"
        )
        graph = make_graph([allReduceMax], "allReduceMax", [input], [output])
        model = make_model(graph)
        from_onnx(model, backend.cpu_runtime())

    def test_allReduceAvg(self):
        input = make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 2, 4])
        output = make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 2, 4])
        allReduceAvg = make_node(
            "AllReduceAvg", ["input"], ["output"], name="allReduceAvg"
        )
        graph = make_graph([allReduceAvg], "allReduceAvg", [input], [output])
        model = make_model(graph)
        from_onnx(model, backend.cpu_runtime())

    def test_split(self):
        input = make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 2, 4])
        split = make_node("Split", ["input"], ["output"], name="split", axis=0)
        output = make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 2, 4])
        make_and_import_model(make_graph([split], "split", [input], [output]))

    def test_split1(self):
        input = make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 2, 4])
        splitAttr = make_tensor("split", TensorProto.INT64, [2], [2, 1])
        output1 = make_tensor_value_info("output1", TensorProto.FLOAT, [1, 2, 2, 4])
        output2 = make_tensor_value_info("output2", TensorProto.FLOAT, [1, 1, 2, 4])
        split = make_node(
            "Split", ["input", "split"], ["output1", "output2"], name="split", axis=1
        )
        make_and_import_model(
            make_graph([split], "split", [input], [output1, output2], [splitAttr])
        )

    def test_allBroadcast(self):
        input = make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 2, 4])
        output = make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 2, 4])
        broadcast = make_node(
            "Broadcast", ["input"], ["output"], name="broadcast", root=1
        )
        graph = make_graph([broadcast], "broadcast", [input], [output])
        model = make_model(graph)
        from_onnx(model, backend.cpu_runtime())

    def test_allGather(self):
        input = make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 2, 4])
        world_size = make_tensor_value_info("world_size", TensorProto.INT32, [1])
        allGather = make_node(
            "AllGather", ["input", "world_size"], ["output"], name="allGather"
        )
        graph = make_graph([allGather], "allGather", [input, world_size], [])
        model = make_model(graph)
        from_onnx(model, backend.cpu_runtime())

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
        a = handler.tensor([1, 2, 3], 12)
        b = handler.tensor([1, 2, 3], 12)
        c = handler.tensor([1, 2, 3], 12)
        d = handler.tensor([1, 2, 3], 12)
        e = handler.tensor([1, 2, 3], 12)

        x = handler.add(
            handler.add(handler.add(handler.add(a, b, None), c, None), d, None), e, None
        )
        y = handler.tensor([3, 2, 1], 12)
        handler.reshape(x, y, [3, 2, 1])

    def test_cast(self):
        input1 = make_tensor_value_info("input1", TensorProto.FLOAT, [1, 3, 2, 4])
        output = make_tensor_value_info("output", TensorProto.FLOAT16, [1, 3, 2, 4])
        cast = make_node(
            "Cast", ["input1"], ["output"], to=TensorProto.FLOAT16, name="cast"
        )
        make_and_import_model(make_graph([cast], "cast", [input1], [output]))

    def test_expand(self):
        data = make_tensor_value_info("data", TensorProto.FLOAT, [3, 1])
        dim = make_tensor_value_info("dim", TensorProto.INT64, [3])
        dim_data = make_tensor("dim", TensorProto.INT64, [3], [2, 1, 6])
        output = make_tensor_value_info("output", TensorProto.FLOAT, [2, 3, 6])
        expand = make_node("Expand", ["data", "dim"], ["output"], name="expand")
        make_and_import_model(
            make_graph([expand], "expand", [data, dim], [output], [dim_data])
        )

    def test_where(self):
        x = make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 7])
        y = make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 5, 7])
        con = make_tensor_value_info("con", TensorProto.BOOL, [1, 3, 5, 7])
        output = make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 5, 7])
        where = make_node("Where", ["x", "y", "con"], ["output"], name="where")
        make_and_import_model(make_graph([where], "where", [x, y, con], [output]))

    def test_send(self):
        sendInput = make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 5, 7])
        send = make_node("Send", ["input"], [], name="send", source=0, destination=1)
        graph = make_graph([send], "send", [sendInput], [])
        model = make_model(graph)
        from_onnx(model, backend.cpu_runtime())

    def test_recv(self):
        recvOutput = make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 5, 7])
        recv = make_node(
            "Recv",
            [],
            ["output"],
            name="recv",
            source=0,
            destination=1,
            shape=[1, 3, 5, 7],
            dataType=1,
        )
        graph = make_graph([recv], "recv", [], [recvOutput])
        model = make_model(graph)
        from_onnx(model, backend.cpu_runtime())


class TestDynamicTensor(unittest.TestCase):
    def test_dynamic_tensor(self):
        filename = r"resnet18-v2-7.onnx"
        current_path = os.getcwd()
        model_file = "/home/featurize/work/my_infiniTensor/InfiniTensor/resnet18-v2-7.onnx"
        for root, dirs, files in os.walk(current_path):
            if filename in files:
                model_file = os.path.join(root, filename)

        model = OnnxStub(onnx.load(model_file), backend.cpu_runtime())
        output_key = list(model.outputs.keys())[0]
        old_output_shape = model.getShape(output_key)
        self.assertEqual(old_output_shape, ([1, 1000]))
        model.set_input([[5, 3, 224, 224]])
        new_output_shape = model.getShape(output_key)
        self.assertEqual(new_output_shape, ([5, 1000]))


if __name__ == "__main__":
    unittest.main()
