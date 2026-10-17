#!/usr/bin/env python3
"""Model-level accuracy and dynamic-shape acceptance test for M1 hardware CI."""
import argparse
import json
from pathlib import Path

import numpy as np


SHAPES = [(batch, sequence) for batch in (1, 3, 8, 17, 32) for sequence in (2, 17, 64, 127, 255, 511, 512)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--seed", type=int, default=315)
    parser.add_argument("--random-shapes", type=int, default=100)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    import onnxruntime as ort
    from pyinfinitensor.embedding import EmbeddingSession

    rng = np.random.default_rng(args.seed)
    session = EmbeddingSession(args.model_dir)
    reference = ort.InferenceSession(str(session.model_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    shapes = SHAPES + [(int(rng.integers(1, 33)), int(rng.integers(1, 513))) for _ in range(args.random_shapes)]
    results = []
    before_uploads = session.stub.weight_upload_count
    for batch, sequence in shapes:
        inputs = {
            "input_ids": rng.integers(0, 1000, (batch, sequence), dtype=np.int64),
            "attention_mask": np.ones((batch, sequence), dtype=np.int64),
            "token_type_ids": np.zeros((batch, sequence), dtype=np.int64),
        }
        actual = next(iter(session.stub.run_numpy(inputs).values()))
        expected = reference.run(None, inputs)[0]
        normalized_actual = actual / np.linalg.norm(actual, axis=1, keepdims=True)
        normalized_expected = expected / np.linalg.norm(expected, axis=1, keepdims=True)
        cosine = np.sum(normalized_actual * normalized_expected, axis=1)
        np.testing.assert_allclose(normalized_actual, normalized_expected, atol=1e-4, rtol=1e-3)
        if np.min(cosine) < 0.99999:
            raise AssertionError("cosine below threshold for shape {}: {}".format((batch, sequence), cosine.min()))
        results.append({"shape": [batch, sequence], "min_cosine": float(cosine.min())})
    report = {"shapes": len(results), "min_cosine": min(item["min_cosine"] for item in results), "weight_uploads_before": before_uploads, "weight_uploads_after": session.stub.weight_upload_count, "ort_providers": reference.get_providers()}
    if report["weight_uploads_before"] != report["weight_uploads_after"]:
        raise AssertionError("initializer upload occurred during shape changes")
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
