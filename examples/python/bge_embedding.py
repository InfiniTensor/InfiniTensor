#!/usr/bin/env python3
"""Encode local Chinese documents and perform a query against them."""

import argparse

from pyinfinitensor.embedding import EmbeddingSession, cosine_top_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--device", choices=("nvidia", "cambricon", "ascend"), default="nvidia"
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--query", required=True)
    parser.add_argument("--document", action="append", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    session = EmbeddingSession(
        args.model_dir, device=args.device, device_id=args.device_id
    )
    corpus = session.encode(args.document, kind="passage", batch_size=args.batch_size)
    query = session.encode([args.query], kind="query", batch_size=1)[0]
    for index, score in cosine_top_k(
        query, corpus, min(args.top_k, len(args.document))
    ):
        print("{:.6f}\t{}".format(score, args.document[index]))


if __name__ == "__main__":
    main()
