#!/usr/bin/env python3
"""Build an offline knowledge index and retrieve/rerank multiple queries."""
import argparse
import json
from pathlib import Path
import sys

from pyinfinitensor.embedding import EmbeddingSession
from pyinfinitensor.reranking import RerankerSession
from pyinfinitensor.retrieval import KnowledgeIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['build', 'query'])
    parser.add_argument('--embedding-model', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--documents', help='UTF-8 JSONL id/text records for build')
    parser.add_argument('--reranker-model', help='frozen cross-encoder package for query')
    parser.add_argument('--query', help='one query; otherwise read JSONL id/query from stdin')
    parser.add_argument('--device', choices=['nvidia', 'cambricon', 'ascend'], default='nvidia')
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--recall-k', type=int, default=50)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--batching', choices=['input', 'length'], default='length')
    parser.add_argument('--execution-mode', choices=['normal', 'auto', 'graph'], default='normal')
    parser.add_argument('--optimizations', choices=['on', 'off'], default='on')
    args = parser.parse_args()
    if args.command == 'build' and not args.documents:
        parser.error('build requires --documents')
    if args.command == 'query' and not args.reranker_model:
        parser.error('query requires --reranker-model')
    optimize = args.optimizations == 'on'
    embedding = EmbeddingSession(args.embedding_model, device=args.device, device_id=args.device_id,
        max_batch_size=args.batch_size if args.command == 'build' else 1,
        memory_views=optimize, fuse_linear_bias=optimize, execution_mode=args.execution_mode)
    if args.command == 'build':
        rows = [json.loads(line) for line in Path(args.documents).read_text(encoding='utf-8').splitlines() if line.strip()]
        index = KnowledgeIndex.build(rows, embedding, batch_size=args.batch_size)
        index.save(args.index)
        print(json.dumps(index.metadata, ensure_ascii=False))
        return
    index = KnowledgeIndex.load(args.index)
    reranker = RerankerSession(args.reranker_model, device=args.device, device_id=args.device_id,
        max_batch_size=args.batch_size, batching=args.batching, memory_views=optimize,
        fuse_linear_bias=optimize, execution_mode=args.execution_mode)
    queries = [{'id': 'query', 'query': args.query}] if args.query is not None else (
        json.loads(line) for line in sys.stdin if line.strip())
    for item in queries:
        if not isinstance(item, dict) or not isinstance(item.get('query'), str):
            raise ValueError('query JSONL requires a string query')
        result = index.search(item['query'], embedding, reranker, args.recall_k, args.top_k)
        print(json.dumps({'id': item.get('id'), **result}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
