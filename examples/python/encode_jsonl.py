#!/usr/bin/env python3
"""Encode local id/text JSONL documents into an offline vector artifact."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pyinfinitensor.embedding import EmbeddingSession


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input", required=True, help="JSONL records with string id/text")
    parser.add_argument("--output", required=True, help="output .npz")
    parser.add_argument("--device", choices=("nvidia", "cambricon", "ascend"), default="nvidia")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--execution-mode", choices=("normal", "auto", "graph"), default="normal")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(row.get("id"), str) or not isinstance(row.get("text"), str) for row in rows):
        raise ValueError("every JSONL record must contain string id and text")
    ids, texts = [row["id"] for row in rows], [row["text"] for row in rows]
    session = EmbeddingSession(args.model_dir, device=args.device, device_id=args.device_id,
                               execution_mode=args.execution_mode)
    vectors = session.encode(texts, kind="passage", batch_size=args.batch_size)
    metadata = {"model_onnx_sha256": sha256(session.model_path), "rows": len(rows), "dimension": int(vectors.shape[1]),
                "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(), "kind": "passage"}
    np.savez_compressed(args.output, ids=np.asarray(ids), vectors=vectors, metadata=json.dumps(metadata, ensure_ascii=False))
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
