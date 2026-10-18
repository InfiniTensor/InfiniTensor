"""Synthetic fixtures test store contracts, not application retrieval quality."""
import tempfile
from pathlib import Path
import unittest
import numpy as np

from infinitensor_evidence import EvidenceStore
from infinitensor_evidence.text import character_chunks,terms


def record(identity='npu',**overrides):
    value={'id':identity,'namespace':'backend','title':'运行图排障','text':'aclrtStreamBeginCapture 图捕获失败，检查运行环境。',
           'source_uri':'repo://docs/runtime.md','revision':'revision-1','platforms':['ascend']}
    value.update(overrides);return value


class TestEvidenceStore(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'evidence.db'
        self.store=EvidenceStore(self.path)

    def tearDown(self):
        self.store.close();self.temp.cleanup()

    def search(self,query='图捕获',**kwargs):
        return self.store.lexical_search(query,'alice',['backend'],**kwargs)

    def test_literal_and_chinese_search_returns_exact_source_span(self):
        value=record();self.store.upsert([value])
        hits=self.search('aclrtStreamBeginCapture')
        self.assertEqual(len(hits),1)
        self.assertEqual(hits[0]['body'],value['text'][hits[0]['start']:hits[0]['end']])
        self.assertEqual(hits[0]['source_revision'],'revision-1')
        self.assertTrue(self.search('捕获失败'))
        self.assertEqual(terms('TS-999'),terms('ts-999'))

    def test_access_platform_and_version_filters_precede_limit(self):
        self.store.upsert([record('secret',readers=['bob']),record('cuda',platforms=['nvidia']),
            record('old',versions={'InfiniOps':['old']}),record('wanted',versions={'InfiniOps':['new']})])
        hits=self.search(filters={'platform':'ascend','versions':{'InfiniOps':'new'}},limit=1)
        self.assertEqual([x['document_id'] for x in hits],['wanted'])
        self.assertEqual(self.store.eligible_chunks('alice',['different']),[])
        with self.assertRaises(ValueError):self.search(filters={'tenant':'other'})
        with self.assertRaises(ValueError):self.search(filters={'platform':'unknown'})

    def test_updates_keep_immutable_old_citations_and_remove_old_search_hits(self):
        self.store.upsert([record()]);old=self.search()[0]['citation_id']
        self.store.upsert([record(text='新文档说明缓存生命周期。',revision='revision-2')])
        self.assertFalse(self.search('aclrtStreamBeginCapture'))
        new=self.search('缓存生命周期')[0]['citation_id']
        self.assertNotEqual(old,new)
        old_value=self.store.read(old,'alice',['backend'])
        self.assertFalse(old_value['is_current']);self.assertEqual(old_value['status'],'superseded')
        self.assertTrue(self.store.read(new,'alice',['backend'])['is_current'])

    def test_revocation_and_withdrawal_also_protect_historical_citations(self):
        self.store.upsert([record()]);old=self.search()[0]['citation_id']
        self.store.upsert([record(readers=['bob'])])
        self.assertFalse(self.search())
        with self.assertRaises(KeyError):self.store.read(old,'alice',['backend'])
        self.assertTrue(self.store.read(old,'bob',['backend']))
        self.store.withdraw('backend','npu')
        with self.assertRaises(KeyError):self.store.read(old,'bob',['backend'])

    def test_failed_encode_does_not_publish_update_or_destroy_previous_index(self):
        def encode(texts):return np.tile(np.asarray([[1.,0.]],np.float32),(len(texts),1))
        self.store.upsert([record()],encode,'frozen-model')
        old=self.search()[0]['citation_id']
        def fail(texts):raise RuntimeError('injected encoder failure')
        with self.assertRaises(RuntimeError):self.store.upsert([record(text='should never be published')],fail,'frozen-model')
        self.assertEqual(self.search()[0]['citation_id'],old)
        with self.assertRaisesRegex(ValueError,'package'):
            self.store.upsert([record(text='different model')],encode,'another-model')
        with self.assertRaisesRegex(ValueError,'normalized'):
            self.store.upsert([record(text='invalid embedding')],lambda t:np.ones((len(t),2)),'frozen-model')
        self.assertEqual(self.search()[0]['citation_id'],old)

    def test_repeated_update_is_noop_and_reopen_is_offline(self):
        self.assertEqual(self.store.upsert([record()])['updated_documents'],1)
        self.assertEqual(self.store.upsert([record()])['updated_documents'],0)
        read_only=EvidenceStore(self.path,read_only=True)
        try:
            self.assertEqual(len(read_only.lexical_search('图捕获','alice',['backend'])),1)
            with self.assertRaises(PermissionError):read_only.withdraw('backend','npu')
        finally:read_only.close()

    def test_invalid_batch_does_not_partially_publish(self):
        with self.assertRaises(ValueError):self.store.upsert([record(),record()])
        with self.assertRaises(ValueError):self.store.upsert([record(),record('bad',revision='')])
        self.assertFalse(self.search())
        with self.assertRaises(ValueError):self.store.upsert([record()],chunker=lambda x:[(0,len(x)+1)])
        self.assertFalse(self.search())

    def test_chunks_cover_original_text_with_overlap(self):
        text='中文原文和错误码 TS-999。'*100
        spans=character_chunks(text,size=80,overlap=15)
        self.assertEqual(spans[0][0],0);self.assertEqual(spans[-1][1],len(text))
        self.assertTrue(all(a[1]-15==b[0] for a,b in zip(spans,spans[1:])))


if __name__=='__main__':unittest.main()
