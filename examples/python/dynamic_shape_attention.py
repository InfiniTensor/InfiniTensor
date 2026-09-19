"""Reproducible PyTorch attention export for dynamic-shape CPU regression.

Usage: python examples/python/dynamic_shape_attention.py real_attention.onnx
"""

import argparse
import hashlib
import io
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

DIM = 32
HEADS = 4
EXPORT_SEED = 0
INPUT_SEED = (2026, 9, 7)
SHAPES = ((2, 5), (1, 1), (2, 7), (3, 16), (1, 3), (2, 5))
PINNED_BATCHES = (2, 1, 8, 3, 2)
PINNED_SEQ = 5


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(DIM, DIM * 3)
        self.out = nn.Linear(DIM, DIM)

    def forward(self, x):
        batch, seq, _ = x.shape
        per_head = DIM // HEADS
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(batch, seq, HEADS, per_head).transpose(1, 2)
        k = k.reshape(batch, seq, HEADS, per_head).transpose(1, 2)
        v = v.reshape(batch, seq, HEADS, per_head).transpose(1, 2)
        attention = (q @ k.transpose(-2, -1)) / (per_head**0.5)
        attention = attention.softmax(dim=-1)
        y = (attention @ v).transpose(1, 2).reshape(batch, seq, DIM)
        return self.out(y)


def export_attention_model():
    """Export in memory, without changing the caller's CPU random state."""
    buffer = io.BytesIO()
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.default_generator.manual_seed(EXPORT_SEED)
        model = Attention().eval()
        torch.onnx.export(
            model,
            (torch.randn(2, 5, DIM),),
            buffer,
            input_names=["x"],
            output_names=["y"],
            dynamic_axes={
                "x": {0: "batch", 1: "seq"},
                "y": {0: "batch", 1: "seq"},
            },
            opset_version=18,
            dynamo=False,
        )
    exported = onnx.load_model_from_string(buffer.getvalue())
    onnx.checker.check_model(exported)
    return exported


def attention_input(batch, seq):
    rng = np.random.default_rng([*INPUT_SEED, batch, seq])
    return rng.standard_normal((batch, seq, DIM)).astype(np.float32)


def export_metadata(model):
    return {
        "model": "real_attention",
        "sha256": hashlib.sha256(model.SerializeToString()).hexdigest(),
        "nodes": len(model.graph.node),
        "operators": dict(sorted(Counter(n.op_type for n in model.graph.node).items())),
        "opsets": {op.domain: op.version for op in model.opset_import},
        "dynamic_axes": {"x": {0: "batch", 1: "seq"}, "y": {0: "batch", 1: "seq"}},
        "hidden_size": DIM,
        "heads": HEADS,
        "export_seed": EXPORT_SEED,
        "example_shape": [2, 5, DIM],
        "dynamo": False,
        "input_seed": [*INPUT_SEED, "batch", "seq"],
        "shapes": SHAPES,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "numpy": np.__version__,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    model = export_attention_model()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, args.output)
    print(json.dumps(export_metadata(model), sort_keys=True))


if __name__ == "__main__":
    main()
