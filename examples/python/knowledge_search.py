"""Build a local document index and retrieve/rerank its candidate passages."""
import argparse
import json

from pyinfinitensor.embedding import EmbeddingSession
from pyinfinitensor.reranking import RerankerSession
from pyinfinitensor.retrieval import build_index, search_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--device", default="nvidia", choices=["nvidia", "cambricon", "ascend"])
    parser.add_argument("--device-id", type=int, default=0)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--documents", required=True, help="JSONL with id/text")
    build.add_argument("--index", required=True)
    query = commands.add_parser("query")
    query.add_argument("--index", required=True)
    query.add_argument("--reranker-model", required=True)
    query.add_argument("--text", required=True)
    query.add_argument("--recall-k", type=int, default=50)
    query.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    embedding = EmbeddingSession(args.embedding_model, args.device, args.device_id)
    if args.command == "build":
        result = build_index(embedding, args.documents, args.index)
    else:
        if not 1 <= args.top_k <= args.recall_k:
            parser.error("require 1 <= top-k <= recall-k")
        candidates = search_index(embedding, args.index, args.text, args.recall_k)
        reranker = RerankerSession(args.reranker_model, args.device, args.device_id)
        ranked = reranker.rerank(args.text, [row["text"] for row in candidates],
                                 args.top_k, ids=[str(row["id"]) for row in candidates])
        result = [dict(candidates[row["index"]], rerank_score=row["score"]) for row in ranked]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
