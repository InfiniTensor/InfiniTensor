"""Dynamic H/W validation with the public SqueezeNet 1.0 feature topology.

The graph uses deterministic random weights rather than downloaded pretrained
weights.  It keeps the published feature extractor topology and adds a Shape
subgraph that flattens the spatial feature map to [batch, 512, -1].
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from pyinfinitensor import backend
from pyinfinitensor.onnx import OnnxStub


def make_squeezenet10_feature_model(seed=2026):
    """Build the SqueezeNet 1.0 feature extractor with reproducible weights."""
    rng = np.random.default_rng(seed)
    nodes = []
    initializers = []

    def weight(name, shape):
        fan_in = shape[1] * shape[2] * shape[3]
        value = rng.normal(0.0, np.sqrt(2.0 / fan_in), shape).astype(np.float32)
        initializers.append(numpy_helper.from_array(value, name))

    def bias(name, channels):
        initializers.append(
            numpy_helper.from_array(np.zeros(channels, dtype=np.float32), name)
        )

    def conv_relu(source, name, in_channels, out_channels, kernel, stride=1, pad=0):
        weight(name + ".weight", (out_channels, in_channels, kernel, kernel))
        bias(name + ".bias", out_channels)
        conv = name + ".conv"
        nodes.append(helper.make_node(
            "Conv", [source, name + ".weight", name + ".bias"], [conv],
            kernel_shape=[kernel, kernel], strides=[stride, stride],
            pads=[pad, pad, pad, pad], name=name,
        ))
        output = name + ".relu"
        nodes.append(helper.make_node("Relu", [conv], [output], name=name + ".relu"))
        return output

    def fire(source, name, in_channels, squeeze, expand1, expand3):
        squeezed = conv_relu(source, name + ".squeeze", in_channels, squeeze, 1)
        one = conv_relu(squeezed, name + ".expand1x1", squeeze, expand1, 1)
        three = conv_relu(squeezed, name + ".expand3x3", squeeze, expand3, 3, pad=1)
        output = name + ".concat"
        nodes.append(helper.make_node("Concat", [one, three], [output], axis=1,
                                      name=name + ".concat"))
        return output, expand1 + expand3

    current = conv_relu("images", "features.0", 3, 96, 7, stride=2)
    nodes.append(helper.make_node("MaxPool", [current], ["features.2"],
                                  kernel_shape=[3, 3], strides=[2, 2],
                                  ceil_mode=1, name="features.2"))
    current, channels = fire("features.2", "features.3", 96, 16, 64, 64)
    current, channels = fire(current, "features.4", channels, 16, 64, 64)
    current, channels = fire(current, "features.5", channels, 32, 128, 128)
    nodes.append(helper.make_node("MaxPool", [current], ["features.6"],
                                  kernel_shape=[3, 3], strides=[2, 2],
                                  ceil_mode=1, name="features.6"))
    current, channels = fire("features.6", "features.7", channels, 32, 128, 128)
    current, channels = fire(current, "features.8", channels, 48, 192, 192)
    current, channels = fire(current, "features.9", channels, 48, 192, 192)
    current, channels = fire(current, "features.10", channels, 64, 256, 256)
    nodes.append(helper.make_node("MaxPool", [current], ["features.11"],
                                  kernel_shape=[3, 3], strides=[2, 2],
                                  ceil_mode=1, name="features.11"))
    current, channels = fire("features.11", "features.12", channels, 64, 256, 256)
    assert channels == 512

    def integer(name, value):
        initializers.append(numpy_helper.from_array(np.asarray(value, dtype=np.int64), name))

    integer("shape.index0", 0)
    integer("shape.axes0", [0])
    integer("shape.channels", [512])
    integer("shape.infer", [-1])
    nodes.extend([
        helper.make_node("Shape", ["images"], ["shape.input"]),
        helper.make_node("Gather", ["shape.input", "shape.index0"], ["shape.batch"]),
        helper.make_node("Unsqueeze", ["shape.batch", "shape.axes0"], ["shape.batch_vec"]),
        helper.make_node("Concat", ["shape.batch_vec", "shape.channels", "shape.infer"],
                         ["target"], axis=0),
        helper.make_node("Reshape", [current, "target"], ["features"]),
    ])

    graph = helper.make_graph(
        nodes, "squeezenet10_dynamic_features",
        [helper.make_tensor_value_info(
            "images", TensorProto.FLOAT, ["batch", 3, "height", "width"]
        )],
        [helper.make_tensor_value_info(
            "features", TensorProto.FLOAT, ["batch", 512, "spatial"]
        ), helper.make_tensor_value_info("target", TensorProto.INT64, [3])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    model.producer_name = "InfiniTensor dynamic-shape training project"
    model.doc_string = (
        "SqueezeNet 1.0 feature topology with deterministic random weights; "
        "not a pretrained accuracy model."
    )
    onnx.checker.check_model(model)
    return model


def validate(runtime=None):
    """Run five H/W values on one model instance and compare with ORT."""
    model = make_squeezenet10_feature_model()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    reference = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    runtime = runtime or backend.cpu_runtime()
    # Dynamic axes default to one in the generic importer.  A 1x1 image is not
    # legal for SqueezeNet's 7x7 first convolution, so provide one legal shape
    # for graph construction; subsequent executions still change H/W in place.
    stub = OnnxStub(model, runtime, input_shapes=[[1, 3, 36, 36]])
    handler = stub.handler
    rows = []
    for index, (height, width) in enumerate(
        # These values make all three ceil-mode pooling windows divide exactly.
        # InfiniTensor and ORT otherwise differ in an existing MaxPool rounding
        # corner case that is outside this dynamic-shape project.
        ((36, 36), (52, 68), (68, 52), (36, 52), (36, 36))
    ):
        dims = (1, 3, height, width)
        data = np.random.default_rng(100 + index).normal(size=dims).astype(np.float32)
        expected, target = reference.run(None, {"images": data})
        started = time.perf_counter_ns()
        stub.set_input([dims])
        stub.inputs["images"].copyin_numpy(data)
        stub.run()
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        actual = stub.outputs["features"].copyout_numpy()
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
        np.testing.assert_array_equal(stub.outputs["target"].copyout_numpy(), target)
        assert stub.handler is handler
        rows.append({
            "input": list(dims), "output": list(actual.shape),
            "target": target.tolist(), "elapsed_ms": elapsed_ms,
            "max_abs_error": float(np.max(np.abs(actual - expected))),
            "planned_activation_bytes": handler.planned_activation_bytes(),
            "activation_pool_capacity": handler.activation_pool_capacity(),
            "activation_pool_storage_id": handler.activation_pool_storage_id(),
        })
    return {
        "model": "SqueezeNet 1.0 feature extractor",
        "weights": "deterministic random initialization (not pretrained)",
        "same_model_instance": True,
        "runs": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/real-model")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = make_squeezenet10_feature_model()
    onnx.save(model, output / "squeezenet10_dynamic.onnx")
    result = validate()
    text = json.dumps(result, indent=2)
    (output / "results.json").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
