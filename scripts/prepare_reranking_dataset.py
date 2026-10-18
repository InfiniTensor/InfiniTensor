#!/usr/bin/env python3
"""Freeze a deterministic public Chinese retrieval/reranking application subset."""
import argparse
import hashlib
import json
from pathlib import Path


REVISION = '76631901a18387f85eaa53e5450019b87ad58ef9'
SOURCE_SHA256 = 'ceaf1ba1a4777dbb7aecad86fb8aa9cae7a678af6d711d5529c238e89bb6adff'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--queries', type=int, default=64)
    parser.add_argument('--negatives', type=int, default=64)
    args = parser.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq
    source, output = Path(args.parquet), Path(args.output)
    if args.queries < 1 or args.negatives < 1:
        raise ValueError('queries and negatives must be positive')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError('source dataset hash differs from fixed revision')
    table = pq.read_table(source)
    queries = table['query'].to_pylist()
    positive_lists = table['positive'].to_pylist()
    eligible = [i for i in range(len(queries)) if positive_lists[i]]
    ordered = sorted(eligible, key=lambda i: (queries[i], i))
    if not ordered:
        raise ValueError('dataset has no queries with relevance judgments')
    # Spread the sample over the full query range, rather than overrepresenting
    # one lexicographic prefix. Selection never reads relevance scores.
    count = min(args.queries, len(ordered))
    positions = [i * (len(ordered)-1) // max(count-1, 1) for i in range(count)]
    indices = [ordered[position] for position in positions]
    selected, documents = [], {}
    for index, row in zip(indices, table.take(pa.array(indices)).to_pylist()):
        positives = set(row['positive'])
        negatives = sorted(set(row['negative']) - positives,
                           key=lambda text: hashlib.sha256(text.encode()).hexdigest())[:args.negatives]
        if not positives or any(not isinstance(text, str) for text in positives | set(negatives)):
            raise ValueError('invalid labeled query row')
        positive_ids = []
        for text in sorted(positives | set(negatives)):
            identity = hashlib.sha256(text.encode()).hexdigest()
            documents[identity] = {'id': identity, 'text': text}
            if text in positives:
                positive_ids.append(identity)
        identity = hashlib.sha256((str(index) + '\n' + row['query']).encode()).hexdigest()
        selected.append({'id': identity, 'source_row': index, 'query': row['query'],
                         'positive_ids': sorted(positive_ids)})
    docs = [documents[key] for key in sorted(documents)]
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in [('queries.jsonl', selected), ('documents.jsonl', docs)]:
        (output / name).write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows), encoding='utf-8')
    manifest = {'dataset': 'C-MTEB/T2Reranking', 'revision': REVISION,
        'source_sha256': digest, 'queries': len(selected), 'documents': len(docs),
        'selection': 'exclude rows without positive judgments; evenly spaced indices across remaining queries sorted by (Unicode query, original row); all positives; first N unique negatives by text SHA256; corpus is the deduplicated union',
        'source_rows': len(queries), 'eligible_rows': len(eligible),
        'negatives_per_query': args.negatives,
        'sha256': {name: hashlib.sha256((output/name).read_bytes()).hexdigest()
                   for name in ['queries.jsonl', 'documents.jsonl']},
        'scope': 'Fixed application subset; not a full public benchmark score'}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
