"""Synthetic contract fixtures are separate from the real-query evaluation."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import tempfile
from pathlib import Path
import unittest
import numpy as np

from infinitensor_evidence.service import EvidenceService
from infinitensor_evidence import EvidenceStore
from test_evidence_store import record


class FixtureEngine:
    fingerprint='test-model';tokenizer_name='fixture-characters'
    def count_tokens(self,text):return len(text)
    def encode_query(self,text):return np.asarray([1.,0.],np.float32)
    def score(self,pairs):return np.arange(len(pairs),dtype=np.float32)


class TestEvidenceService(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'evidence.db'
        self.store=EvidenceStore(self.path)

    def tearDown(self):
        self.store.close();self.temp.cleanup()

    def make(self,engine=None):return EvidenceService(self.store,'alice',['backend'],engine)

    def test_hybrid_filters_before_models_and_returns_traceable_budgeted_evidence(self):
        self.store.upsert([record('allowed'),record('secret',readers=['bob']),record('wrong-platform',platforms=['nvidia'])],
            lambda texts:np.tile(np.asarray([[1.,0.]],np.float32),(len(texts),1)),'test-model')
        engine=FixtureEngine();seen=[]
        def score(pairs):seen.extend(pairs);return np.ones(len(pairs),np.float32)
        engine.score=score;service=self.make(engine)
        result=service.search_evidence('图捕获',filters={'platform':'ascend'},context_budget=1000)
        self.assertEqual([c['document_id'] for c in result['citations']],['allowed'])
        self.assertEqual(len(seen),1)
        self.assertLessEqual(len(result['context']),1000)
        read=service.read_evidence(result['citations'][0]['citation_id'])
        self.assertEqual(read['context'],result['context']);self.assertTrue(read['is_current'])

    def test_budget_never_truncates_an_untraceable_half_quote(self):
        self.store.upsert([record()]);service=self.make()
        result=service.search_evidence('图捕获',mode='lexical',rerank=False,context_budget=10)
        self.assertEqual(result['status'],'context_budget_too_small');self.assertFalse(result['context'])
        hit=service.search_evidence('图捕获',mode='lexical',rerank=False)['citations'][0]
        self.assertEqual(service.read_evidence(hit['citation_id'],context_budget=5)['status'],'context_budget_too_small')

    def test_exact_duplicate_sources_do_not_consume_context_twice(self):
        self.store.upsert([record('a'),record('b')]);service=self.make()
        result=service.search_evidence('图捕获',mode='lexical',rerank=False)
        self.assertEqual(result['candidate_count'],2);self.assertEqual(len(result['citations']),1)

    def test_no_hits_and_invalid_input_are_explicit(self):
        service=self.make()
        self.assertEqual(service.search_evidence('图捕获',mode='lexical',rerank=False)['status'],'no_evidence')
        for kwargs in [dict(query=''),dict(query='x',top_k=True),dict(query='x',filters={'principal':'bob'}),dict(query='x',mode='other')]:
            with self.assertRaises(ValueError):service.search_evidence(**kwargs)
        with self.assertRaisesRegex(ValueError,'explicit'):
            service.search_evidence('x',mode='vector')

    def test_wrong_model_identity_is_rejected_before_forward(self):
        self.store.upsert([record()])
        with self.assertRaisesRegex(ValueError,'package'):
            self.make(FixtureEngine()).search_evidence('图捕获')

    def test_model_calls_are_serial_under_concurrent_requests(self):
        self.store.upsert([record()],lambda t:np.tile(np.asarray([[1.,0.]],np.float32),(len(t),1)),'test-model')
        engine=FixtureEngine();active=0;peak=0;guard=threading.Lock()
        def score(pairs):
            nonlocal active,peak
            with guard:active+=1;peak=max(peak,active)
            time.sleep(.01)
            with guard:active-=1
            return np.ones(len(pairs),np.float32)
        engine.score=score;service=self.make(engine)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results=list(pool.map(lambda _:service.search_evidence('图捕获'),range(8)))
        self.assertEqual(peak,1);self.assertTrue(all(r['citations'] for r in results))


if __name__=='__main__':unittest.main()
