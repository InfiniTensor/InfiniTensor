"""Real CUDA checks: FP32 math, nesting, exceptions and precision inheritance."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from pyinfinitensor.embedding import EmbeddingSession, fp32_inference_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    session = EmbeddingSession(args.model_dir)
    baseline = session.encode(["测试精度继承", "mixed English"])
    rng = np.random.default_rng(328)
    a, b = [rng.normal(size=(64, 64)).astype(np.float32) for _ in range(2)]
    expected = (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)
    x, y = torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda()
    results = []
    nodes = [
        torch.backends.cuda.matmul,
        torch.backends.mkldnn.matmul,
        torch.backends.cudnn.conv,
        torch.backends.cudnn.rnn,
    ]
    for setting in ("highest", "high", "medium"):
        torch.set_float32_matmul_precision(setting)
        before = [node.fp32_precision for node in nodes]
        actual = session.encode(["测试精度继承", "mixed English"])
        np.testing.assert_allclose(actual, baseline, atol=1e-4, rtol=1e-3)
        assert torch.get_float32_matmul_precision() == setting
        assert before == [node.fp32_precision for node in nodes]
        results.append(
            {
                "legacy_policy": setting,
                "embedding_max_abs": float(np.max(np.abs(actual - baseline))),
            }
        )
    for parent in ("ieee", "tf32"):
        torch.backends.fp32_precision = parent
        torch.backends.cudnn.fp32_precision = "none"
        torch.backends.mkldnn.fp32_precision = "none"
        for node in nodes:
            node.fp32_precision = "none"
        before = [node.fp32_precision for node in nodes]
        try:
            with fp32_inference_policy():
                with fp32_inference_policy():
                    np.testing.assert_allclose(
                        (x @ y).cpu().numpy(), expected, atol=2e-5, rtol=1e-5
                    )
                np.testing.assert_allclose(
                    (x @ y).cpu().numpy(), expected, atol=2e-5, rtol=1e-5
                )
                raise RuntimeError("inference failed")
        except RuntimeError as error:
            assert str(error) == "inference failed"
        assert before == [node.fp32_precision for node in nodes]
        for changed in ("tf32", "ieee"):
            torch.backends.fp32_precision = changed
            assert all(node.fp32_precision == changed for node in nodes)
        results.append(
            {
                "inherited_parent": parent,
                "parent_changes_after_exit": "passed",
                "nested_fp32_cuda_matmul_and_exception": "passed",
            }
        )
    from pyinfinitensor.retrieval import build_index, search_index
    import tempfile

    texts = ["中文文本", "包含 English 的更长文本", "", "不同长度", "文档检索"]
    expected_vectors = session.encode(texts)
    for limit in (1, 4):
        small = EmbeddingSession(args.model_dir, max_batch_size=limit)
        np.testing.assert_allclose(
            small.encode(texts), expected_vectors, atol=1e-4, rtol=1e-3
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            documents = root / "documents.jsonl"
            documents.write_text(
                json.dumps({"id": "a", "text": texts[0]}, ensure_ascii=False),
                encoding="utf-8",
            )
            build_index(small, documents, root / "index")
            assert search_index(small, root / "index", "中文文本")[0]["id"] == "a"
        results.append({"max_batch_size": limit, "real_encode_index_query": "passed"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"status": "passed", "torch": torch.__version__, "cases": results}, indent=2
        )
    )
    print(args.output.read_text())


if __name__ == "__main__":
    main()
