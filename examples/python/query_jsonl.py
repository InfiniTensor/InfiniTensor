#!/usr/bin/env python3
"""Query an offline vector artifact produced by encode_jsonl.py."""
import argparse
import json
import numpy as np

from pyinfinitensor.embedding import EmbeddingSession, cosine_top_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--vectors", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", choices=("nvidia", "cambricon", "ascend"), default="nvidia")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--execution-mode", choices=("normal", "auto", "graph"), default="normal")
    args = parser.parse_args()
    artifact = np.load(args.vectors, allow_pickle=False)
    ids, vectors = artifact["ids"], artifact["vectors"]
    metadata = json.loads(str(artifact["metadata"]))
    session = EmbeddingSession(args.model_dir, device=args.device, device_id=args.device_id,
                               execution_mode=args.execution_mode)
    query = session.encode([args.query], kind="query", batch_size=1)[0]
    for index, score in cosine_top_k(query, vectors, min(args.top_k, len(ids)), ids=ids):
        print(json.dumps({"id": str(ids[index]), "score": score}, ensure_ascii=False))
    print(json.dumps({"artifact": metadata}, ensure_ascii=False), file=__import__("sys").stderr)


if __name__ == "__main__":
    main()
