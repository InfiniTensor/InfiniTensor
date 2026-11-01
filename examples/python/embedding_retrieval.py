"""Build and query an offline JSONL document embedding index."""

import argparse
import json
from pyinfinitensor.embedding import EmbeddingSession
from pyinfinitensor.retrieval import build_index, search_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="nvidia")
    parser.add_argument("--device-id", type=int, default=0)
    commands = parser.add_subparsers(dest="command", required=True)
    encode = commands.add_parser("encode")
    encode.add_argument(
        "--documents", required=True, help="UTF-8 JSONL containing id/text"
    )
    encode.add_argument("--output", required=True, help="new index directory")
    encode.add_argument("--batch-size", type=int, default=32)
    query = commands.add_parser("query")
    query.add_argument("--index", required=True)
    query.add_argument("--text", required=True)
    query.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    session = EmbeddingSession(args.model_dir, args.device, args.device_id)
    result = (
        build_index(session, args.documents, args.output, args.batch_size)
        if args.command == "encode"
        else search_index(session, args.index, args.text, args.top_k)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
