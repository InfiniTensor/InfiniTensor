import unittest
import numpy as np

from infinitensor_evidence.engine import InferenceCache


class TestInferenceCache(unittest.TestCase):
    def test_exact_pairs_deduplicate_preserve_order_and_survive_eviction(self):
        calls=[]
        def score(pairs):calls.append(pairs);return np.asarray([len(p[1]) for p in pairs],np.float32)
        cache=InferenceCache(lambda q:np.array([1.,0.]),score,1,1)
        pairs=[('q','a'),('q','bbbb'),('q','a')]
        np.testing.assert_array_equal(cache.score(pairs),[1,4,1])
        self.assertEqual(len(calls[0]),2);self.assertEqual(len(cache.pairs),1)
        np.testing.assert_array_equal(cache.score([('q','bbbb')]),[4])
        self.assertEqual(len(calls),1)
        cache.score([('new query','bbbb')]);self.assertEqual(len(calls),2)

    def test_cached_embedding_outputs_own_their_memory(self):
        calls=[]
        def encode(text):calls.append(text);return np.asarray([1.,0.],np.float32)
        cache=InferenceCache(encode,lambda p:np.ones(len(p)),1,0)
        value=cache.query('q');value[0]=0
        np.testing.assert_array_equal(cache.query('q'),[1,0]);self.assertEqual(calls,['q'])
        cache.query('other');cache.query('q');self.assertEqual(len(calls),3)

    def test_zero_capacity_recomputes_and_failures_do_not_cache(self):
        calls=[]
        def score(pairs):calls.append(pairs);return np.ones(len(pairs))
        cache=InferenceCache(lambda q:np.array([1.,0.]),score,0,0)
        cache.score([('q','p')]);cache.score([('q','p')]);self.assertEqual(len(calls),2)
        cache.pair_scorer=lambda p:np.asarray([np.nan]*len(p))
        with self.assertRaises(ValueError):cache.score([('q','p')])
        self.assertFalse(cache.pairs)


if __name__=='__main__':unittest.main()
