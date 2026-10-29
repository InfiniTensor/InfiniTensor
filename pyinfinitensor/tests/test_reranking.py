import unittest

import numpy as np

from pyinfinitensor.reranking import RerankerSession, position_ids


class Tokenizer:
    pad_token_id = 1

    def __call__(self, queries, documents, **kwargs):
        ids = [[0, int(query) + 3] + [7] * len(document) + [2]
               for query, document in zip(queries, documents)]
        return {"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]}


class Stub:
    last_timing = dict(prepare_s=0.0, h2d_s=0.0, run_s=0.0, d2h_s=0.0)

    def __init__(self):
        self.shapes = []
        self.dtype = np.float32

    def run_numpy(self, inputs):
        self.shapes.append(inputs["input_ids"].shape)
        np.testing.assert_array_equal(inputs["position_ids"], position_ids(inputs["input_ids"]))
        return {"scores": inputs["input_ids"][:, 1:2].astype(self.dtype)}


class TestReranking(unittest.TestCase):
    def session(self):
        value = RerankerSession.__new__(RerankerSession)
        value.max_batch_size, value.max_length, value.token_budget = 2, 512, 16
        value.batching, value.tokenizer, value.stub = "length", Tokenizer(), Stub()
        value.last_timing = {}
        return value

    def test_exact_lengths_batching_preserves_pair_order(self):
        session = self.session()
        result = session.score([("1", "abcdef"), ("2", "a"), ("3", "abc")])
        np.testing.assert_array_equal(result, [4, 5, 6])
        self.assertEqual(session.stub.shapes, [(2, 6), (1, 9)])
        saved = result.copy()
        session.score([("9", "a")])
        np.testing.assert_array_equal(result, saved)

    def test_empty_and_invalid_inputs(self):
        session = self.session()
        self.assertEqual(session.score([]).shape, (0,))
        self.assertEqual(session.stub.shapes, [])
        for pairs in [[("1",)], [("1", 3)]]:
            with self.assertRaises(TypeError):
                session.score(pairs)
        with self.assertRaises(ValueError):
            session.score([("1", "a")], batch_size=True)

    def test_device_output_must_remain_fp32(self):
        session = self.session()
        session.stub.dtype = np.float16
        with self.assertRaises(RuntimeError):
            session.score([("1", "a")])


if __name__ == "__main__":
    unittest.main()
