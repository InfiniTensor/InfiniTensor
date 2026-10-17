#!/usr/bin/env python3
"""Freeze BAAI/bge-small-zh-v1.5 as a dynamic-opset-17 CLS ONNX package."""
import argparse
import hashlib
import json
from pathlib import Path
import platform


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--revision", required=True, help="immutable Hugging Face commit SHA")
    parser.add_argument("--source", default="BAAI/bge-small-zh-v1.5",
                        help="Hub model id or an already verified local model package")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    import onnx
    import torch
    from transformers import AutoModel, AutoTokenizer, __version__ as transformers_version

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.source)
    local_source = source.is_dir()
    load_args = {"local_files_only": True} if local_source else {"revision": args.revision}
    tokenizer = AutoTokenizer.from_pretrained(str(source) if local_source else args.source, **load_args)
    torch.manual_seed(315)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModel.from_pretrained(
        str(source) if local_source else args.source,
        attn_implementation="eager", **load_args,
    ).eval()
    model.config.use_cache = False
    tokenizer.save_pretrained(output)
    # Keep the reference weights beside the ONNX artifact so the same frozen
    # package can be used for PyTorch accuracy/performance baselines offline.
    model.save_pretrained(output, safe_serialization=True)

    class ClsEncoder(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state[:, 0, :]

    sample = tokenizer(["动态输入", "中文 embedding"], padding=True, return_tensors="pt")
    for name in ("input_ids", "attention_mask"):
        sample[name] = sample[name].to(torch.int64)
    sample["token_type_ids"] = sample.get("token_type_ids", torch.zeros_like(sample["input_ids"])).to(torch.int64)
    path = output / "model.onnx"
    # Export restores the wrapper's training state afterwards.  Set the
    # wrapper itself to eval, otherwise it restores dropout in the encoder.
    wrapper = ClsEncoder(model).eval()
    torch.onnx.export(
        wrapper, (sample["input_ids"], sample["attention_mask"], sample["token_type_ids"]), str(path),
        input_names=["input_ids", "attention_mask", "token_type_ids"], output_names=["embeddings"],
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "attention_mask": {0: "batch", 1: "sequence"}, "token_type_ids": {0: "batch", 1: "sequence"}, "embeddings": {0: "batch"}},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    onnx.checker.check_model(str(path))
    artifact_hashes = {
        str(item.relative_to(output)): sha256(item)
        for item in output.rglob("*") if item.is_file() and item.name != "manifest.json"
    }
    manifest = {
        "model": "BAAI/bge-small-zh-v1.5", "revision": args.revision, "opset": 17,
        "source": str(source.resolve()) if local_source else args.source,
        "pooling": "CLS", "normalization": "L2 outside graph", "sha256": artifact_hashes,
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers_version, "onnx": onnx.__version__},
        "nodes": [{"name": node.name, "op": node.op_type,
                   "inputs": list(node.input), "outputs": list(node.output)}
                  for node in onnx.load(str(path)).graph.node],
    }
    if args.validate:
        import numpy as np
        import onnxruntime as ort
        with torch.inference_mode():
            reference = wrapper(sample["input_ids"], sample["attention_mask"], sample["token_type_ids"]).numpy()
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        actual = session.run(None, {name: value.numpy() for name, value in sample.items()})[0]
        np.testing.assert_allclose(actual, reference, atol=1e-4, rtol=1e-3)
        manifest["pytorch_ort_validation"] = "passed (CPUExecutionProvider)"
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
