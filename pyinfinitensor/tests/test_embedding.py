import sys
import unittest
import tempfile
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pyinfinitensor.embedding import (
    BGE_QUERY_PREFIX,
    EmbeddingSession,
    cosine_top_k,
    fp32_inference_policy,
    verify_model_package,
)


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return {
            "input_ids": np.ones((len(texts), 3), np.int64),
            "attention_mask": np.ones((len(texts), 3), np.int64),
        }


class FakeStub:
    def __init__(self):
        self.last_timing = {"prepare_s": 0.0, "h2d_s": 0.0, "run_s": 0.0, "d2h_s": 0.0}

    def run_numpy(self, inputs):
        batch = inputs["input_ids"].shape[0]
        output = np.zeros((batch, 512), np.float32)
        output[:, 0] = np.arange(1, batch + 1)
        output[:, 1] = 1
        return {"embeddings": output}


class TestEmbeddingSession(unittest.TestCase):
    def make_session(self):
        session = EmbeddingSession.__new__(EmbeddingSession)
        session.max_batch_size, session.max_length = 32, 512
        session.tokenizer, session.stub, session.last_timing = (
            FakeTokenizer(),
            FakeStub(),
            {},
        )
        return session

    def test_query_prefix_batching_and_normalization(self):
        session = self.make_session()
        vectors = session.encode(["甲", "乙", "丙"], kind="query", batch_size=2)
        self.assertEqual(vectors.shape, (3, 512))
        self.assertTrue(np.allclose(np.linalg.norm(vectors, axis=1), 1))
        self.assertEqual(
            session.tokenizer.calls[0][0],
            [BGE_QUERY_PREFIX + "甲", BGE_QUERY_PREFIX + "乙"],
        )
        self.assertEqual(session.tokenizer.calls[1][0], [BGE_QUERY_PREFIX + "丙"])

    def test_precision_settings_restore_after_inference_error_and_partial_provider(
        self,
    ):
        flag = SimpleNamespace(allow_tf32=True)
        fake_torch = SimpleNamespace(
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=flag)),
            npu=SimpleNamespace(matmul=SimpleNamespace(allow_hf32=True)),
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                with fp32_inference_policy():
                    self.assertFalse(flag.allow_tf32)
                    self.assertFalse(fake_torch.npu.matmul.allow_hf32)
                    raise RuntimeError("inference failed")
        self.assertTrue(flag.allow_tf32)
        self.assertTrue(fake_torch.npu.matmul.allow_hf32)

    def test_real_torch_precision_state_is_preserved(self):
        import torch

        previous = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("medium")
            with fp32_inference_policy():
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                self.assertFalse(torch.backends.cudnn.allow_tf32)
                modern = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
                if modern is not None:
                    self.assertEqual(modern, "ieee")
                else:
                    self.assertEqual(torch.get_float32_matmul_precision(), "highest")
            self.assertEqual(torch.get_float32_matmul_precision(), "medium")
        finally:
            torch.set_float32_matmul_precision(previous)

    def test_new_precision_api_preserves_backend_overrides(self):
        import torch

        matmul = torch.backends.cuda.matmul
        if not hasattr(matmul, "fp32_precision"):
            return  # Older supported PyTorch uses the legacy policy tested above.
        previous = matmul.fp32_precision
        try:
            for setting in ("tf32", "ieee", "none"):
                matmul.fp32_precision = setting
                with self.assertRaisesRegex(RuntimeError, "inference failed"):
                    with fp32_inference_policy():
                        self.assertEqual(matmul.fp32_precision, "ieee")
                        self.assertEqual(
                            torch.backends.cudnn.conv.fp32_precision, "ieee"
                        )
                        raise RuntimeError("inference failed")
                self.assertEqual(matmul.fp32_precision, setting)
        finally:
            matmul.fp32_precision = previous

    def test_local_package_integrity_is_checked_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / "model.onnx"
            model.write_bytes(b"frozen model bytes")
            manifest = {
                "model": "BAAI/bge-small-zh-v1.5",
                "revision": "a" * 40,
                "sha256": {
                    "model.onnx": hashlib.sha256(model.read_bytes()).hexdigest()
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(verify_model_package(root, model), manifest)
            (root / "tokenizer_config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "unrecorded tokenizer"):
                verify_model_package(root, model)
            (root / "tokenizer_config.json").unlink()
            model.write_bytes(b"modified model")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_model_package(root, model)

    def test_deterministic_retrieval(self):
        result = cosine_top_k(np.array([1, 0]), np.array([[1, 0], [1, 0], [0, 1]]), 2)
        self.assertEqual([item[0] for item in result], [0, 1])
        by_id = cosine_top_k(
            np.array([1, 0]),
            np.array([[1, 0], [1, 0], [0, 1]]),
            2,
            ids=np.array(["doc-z", "doc-a", "doc-m"]),
        )
        self.assertEqual([item[0] for item in by_id], [1, 0])

    def test_empty_input_and_invalid_batches(self):
        session = self.make_session()
        self.assertEqual(session.encode([]).shape, (0, 512))
        self.assertEqual(session.tokenizer.calls, [])
        for batch in (0, 33, 1.5, True, "2"):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                session.encode(["甲"], batch_size=batch)
        with self.assertRaises(TypeError):
            session.encode(["甲", 1])

    def test_fp32_policy_restores_every_provider_flag(self):
        def flag(value):
            return SimpleNamespace(allow_tf32=value)

        fake_torch = SimpleNamespace(
            backends=SimpleNamespace(
                cuda=SimpleNamespace(matmul=flag(True)),
                cudnn=flag(True),
                mlu=SimpleNamespace(matmul=flag(True), cnnl=flag(False)),
            ),
            npu=SimpleNamespace(
                config=SimpleNamespace(allow_internal_format=True),
                conv=SimpleNamespace(allow_hf32=True),
                matmul=SimpleNamespace(allow_hf32=False),
            ),
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with fp32_inference_policy():
                self.assertFalse(fake_torch.backends.cuda.matmul.allow_tf32)
                self.assertFalse(fake_torch.backends.cudnn.allow_tf32)
                self.assertFalse(fake_torch.backends.mlu.matmul.allow_tf32)
                self.assertFalse(fake_torch.backends.mlu.cnnl.allow_tf32)
                self.assertFalse(fake_torch.npu.config.allow_internal_format)
                self.assertFalse(fake_torch.npu.conv.allow_hf32)
                self.assertFalse(fake_torch.npu.matmul.allow_hf32)
        self.assertTrue(fake_torch.backends.cuda.matmul.allow_tf32)
        self.assertTrue(fake_torch.backends.cudnn.allow_tf32)
        self.assertTrue(fake_torch.backends.mlu.matmul.allow_tf32)
        self.assertFalse(fake_torch.backends.mlu.cnnl.allow_tf32)
        self.assertTrue(fake_torch.npu.config.allow_internal_format)
        self.assertTrue(fake_torch.npu.conv.allow_hf32)
        self.assertFalse(fake_torch.npu.matmul.allow_hf32)


if __name__ == "__main__":
    unittest.main()
