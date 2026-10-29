"""Compare persisted BGE retrieval with same-device PyTorch on a frozen T2 subset."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from validate_bge_embedding import compare, normalized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()
    selected = json.loads(args.selection.read_text(encoding="utf-8"))
    expected = selected.pop("selection_sha256")
    canonical = json.dumps(
        selected, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() == expected
    assert selected["dataset_revision"] == "8731a845f1bf500a4f111cf1070785c793d10e64"
    corpus, queries = selected["corpus"], selected["queries"]
    assert len(corpus) == 1000 and len(queries) == 100
    ids = [d["id"] for d in corpus]
    assert ids == sorted(ids) and [q["id"] for q in queries] == sorted(
        q["id"] for q in queries
    )
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "dataset_revision": selected["dataset_revision"],
        "selection_sha256": expected,
        "scope": "engine migration consistency, not retrieval leaderboard quality",
    }

    def save():
        (args.output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    try:
        import torch
        from transformers import AutoModel
        from pyinfinitensor.embedding import (
            EmbeddingSession,
            BGE_QUERY_PREFIX,
            fp32_inference_policy,
            cosine_top_k,
        )
        from pyinfinitensor.retrieval import build_index, search_index

        session = EmbeddingSession(args.model_dir, "nvidia", args.device_id)
        documents = args.output / "documents.jsonl"
        with documents.open("w", encoding="utf-8") as handle:
            for doc in corpus:
                handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
        build_index(session, documents, args.output / "index")
        actual_corpus = np.load(args.output / "index/vectors.npy", allow_pickle=False)
        actual_queries = session.encode([q["text"] for q in queries], kind="query")
        reference = (
            AutoModel.from_pretrained(
                str(args.model_dir), local_files_only=True, attn_implementation="eager"
            )
            .eval()
            .cuda(args.device_id)
        )

        def encode_reference(texts, kind):
            vectors = []
            for start in range(0, len(texts), 32):
                chunk = texts[start : start + 32]
                if kind == "query":
                    chunk = [BGE_QUERY_PREFIX + text for text in chunk]
                inputs = session.tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                    return_token_type_ids=True,
                )
                with fp32_inference_policy(), torch.inference_mode():
                    outputs = reference(
                        **{k: v.cuda(args.device_id) for k, v in inputs.items()}
                    ).last_hidden_state[:, 0, :]
                    vectors.append(normalized(outputs.cpu().numpy()))
            return np.concatenate(vectors)

        reference_corpus = encode_reference([d["text"] for d in corpus], "passage")
        reference_queries = encode_reference([q["text"] for q in queries], "query")
        report["corpus_accuracy"] = compare(actual_corpus, reference_corpus)
        report["query_accuracy"] = compare(actual_queries, reference_queries)
        actual_top = [
            [ids[i] for i, _ in cosine_top_k(query, actual_corpus, 10, ids)]
            for query in actual_queries
        ]
        reference_top = [
            [ids[i] for i, _ in cosine_top_k(query, reference_corpus, 10, ids)]
            for query in reference_queries
        ]
        overlaps = [
            len(set(a) & set(b)) / 10 for a, b in zip(actual_top, reference_top)
        ]
        report["mean_top10_overlap"] = float(np.mean(overlaps))
        report["per_query"] = [
            {"id": q["id"], "actual": a, "reference": b, "overlap": o}
            for q, a, b, o in zip(queries, actual_top, reference_top, overlaps)
        ]
        assert report["mean_top10_overlap"] >= 0.99
        standalone = search_index(
            session, args.output / "index", queries[0]["text"], 10
        )
        assert [d["id"] for d in standalone] == actual_top[0]
        report["persisted_search"] = "passed"
        report["status"] = "passed"
        save()
        print(
            json.dumps(
                {k: v for k, v in report.items() if k != "per_query"},
                ensure_ascii=False,
                indent=2,
            )
        )
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        save()
        raise


if __name__ == "__main__":
    main()
