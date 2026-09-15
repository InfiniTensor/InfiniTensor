"""Small, deterministic ONNX fixtures; no framework or downloads required."""

import numpy as np
from onnx import TensorProto as T, checker, helper as h, numpy_helper as nh


def batch_model(features=6, outputs=4):
    if features <= 0 or features % 2:
        raise ValueError("features must be a positive even integer")
    weights = np.arange(features * outputs, dtype=np.float32).reshape(
        features, outputs
    ) / (features * outputs)
    nodes = [
        h.make_node("Shape", ["x"], ["input_shape"]),
        h.make_node("Gather", ["input_shape", "zero"], ["batch"], axis=0),
        h.make_node("Unsqueeze", ["batch", "axes"], ["batch_vector"]),
        h.make_node("Shape", ["weight"], ["weight_shape"]),
        h.make_node("Gather", ["weight_shape", "zero"], ["features"], axis=0),
        h.make_node("Unsqueeze", ["features", "axes"], ["feature_vector"]),
        h.make_node("Concat", ["batch_vector", "feature_vector"], ["target"], axis=0),
        h.make_node("Reshape", ["x", "target"], ["flat"]),
        h.make_node("MatMul", ["flat", "weight"], ["y"]),
    ]
    initializers = [
        nh.from_array(weights, "weight"),
        nh.from_array(np.array(0, np.int64), "zero"),
        nh.from_array(np.array([0], np.int64), "axes"),
    ]
    model = h.make_model(
        h.make_graph(
            nodes,
            "dynamic_batch",
            [h.make_tensor_value_info("x", T.FLOAT, ["batch", 2, features // 2])],
            [h.make_tensor_value_info("y", T.FLOAT, ["batch", outputs])],
            initializers,
        ),
        opset_imports=[h.make_opsetid("", 13)],
        ir_version=8,
    )
    checker.check_model(model)
    return model
