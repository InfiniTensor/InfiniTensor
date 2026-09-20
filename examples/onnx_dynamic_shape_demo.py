
import argparse

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from pyinfinitensor import backend
from pyinfinitensor.onnx import OnnxStub


def build_model():
    def const(name, value, dtype):
        return numpy_helper.from_array(np.asarray(value, dtype=dtype), name)

    nodes = [
        helper.make_node("Shape", ["x"], ["x_shape"]),
        helper.make_node("Gather", ["x_shape", "index"], ["batch_scalar"], axis=0),
        helper.make_node("Unsqueeze", ["batch_scalar", "axes"], ["batch_vector"]),
        helper.make_node("Cast", ["six_i32"], ["six_i64"], to=TensorProto.INT64),
        helper.make_node("Concat", ["batch_vector", "six_i64"], ["target"], axis=0),
        helper.make_node("Reshape", ["x", "target"], ["flat"]),
        helper.make_node("Add", ["flat", "bias"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "runtime_shape_demo",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 6])],
        initializer=[
            const("index", 0, np.int64),
            const("axes", [0], np.int64),
            const("six_i32", [6], np.int32),
            const("bias", [0.25, -1.0, 2.0, 0.5, -0.5, 3.0], np.float32),
        ],
        value_info=[
            helper.make_tensor_value_info("flat", TensorProto.FLOAT, ["batch", 6])
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=8
    )
    onnx.checker.check_model(model)
    return model


def validate_model(naive=False, batches=(1, 2, 8, 3, 1), verbose=True):
    model = build_model()
    stub = OnnxStub(model, backend.cpu_runtime(), use_naive_allocator=naive)
    reference = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(2026)
    records = []
    for batch in batches:
        x = rng.standard_normal((batch, 2, 3)).astype(np.float32)
        stub.set_input([list(x.shape)])
        stub.inputs["x"].copyin_numpy(x)
        stub.run()

        expected = reference.run(["y"], {"x": x})[0]
        actual_shape = tuple(stub.getShape("y"))
        if actual_shape != expected.shape:
            raise AssertionError(
                f"Shape mismatch: InfiniTensor={actual_shape}, ORT={expected.shape}"
            )
        actual = np.asarray(
            stub.outputs["y"].copyout_float(), dtype=np.float32
        ).reshape(actual_shape)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        max_abs_error = float(np.max(np.abs(actual - expected)))
        records.append((batch, actual_shape, max_abs_error))
        if verbose:
            print(
                f"batch={batch}, shape={actual_shape}, "
                f"max_abs_error={max_abs_error:.8g}, PASS"
            )
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--naive", action="store_true")
    parser.add_argument("--save-model", default=None)
    args = parser.parse_args()
    if args.save_model:
        onnx.save(build_model(), args.save_model)
    validate_model(naive=args.naive)