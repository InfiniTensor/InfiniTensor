import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pyinfinitensor.embedding import (
    BGE_QUERY_PREFIX,
    EmbeddingSession,
    cosine_top_k,
    fp32_inference_policy,
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
