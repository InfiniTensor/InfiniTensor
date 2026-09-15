"""Train a compact digits CNN on a real dataset and export dynamic N/H/W ONNX.

Training/export only needs torch, scikit-learn and onnx. Inference uses the
exported artifact and does not depend on torch or scikit-learn.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
from torch import nn


class DigitsCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.weight = nn.Parameter(torch.randn(32, 10) * 0.1)
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, images):
        features = self.features(images).reshape(images.shape[0], -1)
        return features @ self.weight + self.bias


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("examples/dynamic_shape/artifacts")
    )
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    torch.manual_seed(2026)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    digits = load_digits()
    train, test = train_test_split(
        np.arange(len(digits.target)),
        test_size=0.2,
        stratify=digits.target,
        random_state=2026,
    )
    images = torch.from_numpy(digits.images.astype(np.float32)[:, None] / 16)
    labels = torch.from_numpy(digits.target.astype(np.int64))
    net = DigitsCNN()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
    net.train()
    for epoch in range(args.epochs):
        order = torch.from_numpy(train)[torch.randperm(len(train))]
        for indices in order.split(128):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(net(images[indices]), labels[indices])
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"epoch={epoch + 1}, loss={loss.item():.6f}", flush=True)
    net.eval()
    with torch.no_grad():
        logits = net(images[test])
        accuracy = (logits.argmax(1) == labels[test]).float().mean().item()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "digits_cnn.onnx"
    torch.onnx.export(
        net,
        images[:1],
        str(path),
        input_names=["images"],
        output_names=["logits"],
        opset_version=13,
        dynamo=False,
        dynamic_axes={
            "images": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch"},
        },
    )
    model = onnx.load(path)
    onnx.checker.check_model(model)
    assert any(n.op_type == "Shape" for n in model.graph.node)
    np.savez_compressed(
        args.output_dir / "digits_samples.npz",
        images=images[test[:8]].numpy(),
        labels=labels[test[:8]].numpy(),
        logits=logits[:8].numpy(),
    )
    metadata = {
        "dataset": "scikit-learn load_digits (UCI optical recognition of handwritten digits)",
        "dataset_url": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
        "train_examples": len(train),
        "test_examples": len(test),
        "seed": 2026,
        "epochs": args.epochs,
        "held_out_accuracy": accuracy,
        "parameters": sum(p.numel() for p in net.parameters()),
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "onnx": onnx.__version__,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "operators": [n.op_type for n in model.graph.node],
    }
    (args.output_dir / "training.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
