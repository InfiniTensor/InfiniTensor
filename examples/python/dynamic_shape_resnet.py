"""Standard torchvision ResNet18 with dynamic batch/H/W, for CPU validation.

Uses seeded random weights, not pretrained accuracy evaluation. Example:
    python examples/python/dynamic_shape_resnet.py resnet18_dynamic.onnx
"""

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import onnx
import torch
import torchvision

SHAPES = (
    (1, 3, 32, 32),
    (2, 3, 40, 48),
    (1, 3, 64, 48),
    (3, 3, 32, 64),
    (1, 3, 48, 32),
    (1, 3, 32, 32),
)
EXPORT_SEED = 20260908


def resnet_input(shape):
    return (
        np.random.default_rng([2026, 9, 8, *shape])
        .standard_normal(shape)
        .astype(np.float32)
    )


def export_resnet_model():
    buffer = io.BytesIO()
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.default_generator.manual_seed(EXPORT_SEED)
        model = torchvision.models.resnet18(weights=None).eval()
        torch.onnx.export(
            model,
            (torch.randn(*SHAPES[0]),),
            buffer,
            input_names=["images"],
            output_names=["logits"],
            dynamic_axes={
                "images": {0: "batch", 2: "height", 3: "width"},
                "logits": {0: "batch"},
            },
            opset_version=18,
            dynamo=False,
        )
    exported = onnx.load_model_from_string(buffer.getvalue())
    onnx.checker.check_model(exported)
    return exported


def export_metadata(model):
    import torchvision.models.resnet as source

    return dict(
        architecture="torchvision.models.resnet18",
        weights="None (seeded random)",
        seed=EXPORT_SEED,
        opset=18,
        dynamo=False,
        sha256=hashlib.sha256(model.SerializeToString()).hexdigest(),
        source_sha256=hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest(),
        versions=dict(
            torch=torch.__version__,
            torchvision=torchvision.__version__,
            onnx=onnx.__version__,
            numpy=np.__version__,
        ),
        operators=dict(Counter(n.op_type for n in model.graph.node)),
        nodes=len(model.graph.node),
        shapes=SHAPES,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = export_resnet_model()
    # Preserve existing models and their provenance.
    with args.output.open("xb") as stream:
        stream.write(model.SerializeToString())
    print(json.dumps(export_metadata(model), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
