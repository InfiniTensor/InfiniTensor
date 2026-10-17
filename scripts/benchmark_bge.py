#!/usr/bin/env python3
"""Measure the M1 InfiniTensor embedding path with reproducible inputs."""
import argparse
import json
from statistics import median
import time

import numpy as np


CASES = [(1, 32), (1, 128), (1, 512), (8, 128), (32, 128)]


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    from pyinfinitensor.embedding import EmbeddingSession

    session = EmbeddingSession(args.model_dir)
    report = {"engine": "InfiniTensor", "cases": [], "method": "end-to-end encode; run_s includes host submission and stream synchronization performed by the runtime"}
    for batch, sequence in CASES:
        # Repeated Han characters make tokenizer output deterministic and close
        # to the requested actual length without padding to a static bucket.
        text = "测" * max(1, sequence - 2)
        texts = [text] * batch
        for _ in range(args.warmup):
            session.encode(texts, batch_size=batch)
        samples, segments = [], {key: [] for key in session.last_timing}
        for _round in range(args.rounds):
            for _ in range(args.runs):
                started = time.perf_counter()
                session.encode(texts, batch_size=batch)
                samples.append(time.perf_counter() - started)
                for key, value in session.last_timing.items():
                    segments.setdefault(key, []).append(value)
        p50, p95 = percentile(samples, 50), percentile(samples, 95)
        report["cases"].append({
            "shape": [batch, sequence], "samples": len(samples), "p50_s": p50, "p95_s": p95,
            "texts_per_s": batch / p50, "tokens_per_s": batch * sequence / p50,
            "segments_p50_s": {key: percentile(values, 50) for key, values in segments.items()},
        })
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output == "-":
        print(output)
    else:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(output + "\n")


if __name__ == "__main__":
    main()
