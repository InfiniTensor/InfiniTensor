"""Synthetic fixtures test store contracts, not application retrieval quality."""
import tempfile
from pathlib import Path
import unittest
import subprocess
import sys
import numpy as np

from infinitensor_evidence import EvidenceStore
from infinitensor_evidence.text import character_chunks,terms


def record(identity='npu',**overrides):
    value={'id':identity,'namespace':'backend','title':'运行图排障','text':'aclrtStreamBeginCapture 图捕获失败，检查运行环境。',
           'source_uri':'repo://docs/runtime.md','revision':'revision-1','platforms':['ascend']}
    value.update(overrides);return value


class TestEvidenceStore(unittest.TestCase):
    def test_cli_empty_sync_is_limited_to_host_namespaces(self):
        self.store.upsert([record('protected',namespace='other')])
        empty=Path(self.temp.name)/'empty.jsonl'
        empty.write_text('',encoding='utf-8')
        command=[sys.executable,'-m','infinitensor_evidence','build','--store',str(self.path),
                 '--documents',str(empty),'--principal','alice','--namespace','allowed',
                 '--sync-namespace','other','--lexical-only']
        child=subprocess.run(command,capture_output=True,text=True)
        self.assertNotEqual(child.returncode,0)
        self.assertIn('outside the supplied namespaces',child.stderr)
        self.assertEqual(len(self.store.eligible_chunks('alice',['other'])),1)
        command[command.index('allowed')]='other'
        subprocess.run(command,capture_output=True,text=True,check=True)
        self.assertEqual(self.store.eligible_chunks('alice',['other']),[])

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

    def test_explicit_namespace_sync_withdraws_only_absent_documents(self):
        self.store.upsert([record('keep'),record('remove'),record('other',namespace='different')])
        result=self.store.upsert([record('keep')],sync_namespace='backend')
        self.assertEqual(result['withdrawn_documents'],1)
        self.assertEqual([x['document_id'] for x in self.search()],['keep'])
        self.assertEqual(len(self.store.lexical_search('图捕获','alice',['different'])),1)
        with self.assertRaises(ValueError):self.store.upsert([record(namespace='different')],sync_namespace='backend')

    def test_concurrent_writer_cannot_publish_a_mixed_snapshot(self):
        encode=lambda t:np.tile(np.asarray([[1.,0.]],np.float32),(len(t),1))
        self.store.upsert([record('keep'),record('change')],encode,'model')
        other=EvidenceStore(self.path)
        try:
            def race(texts):
                other.upsert([record('keep',revision='writer-b-2'),record('change',revision='writer-b-3')],encode,'model')
                return encode(texts)
            with self.assertRaisesRegex(RuntimeError,'concurrently'):
                self.store.upsert([record('keep'),record('change',revision='writer-a-2')],race,'model',sync_namespace='backend')
            rows=self.search()
            self.assertEqual({x['document_id']:x['source_revision'] for x in rows},
                             {'keep':'writer-b-2','change':'writer-b-3'})
        finally:other.close()

    def test_read_checks_current_and_historical_environment_scope(self):
        self.store.upsert([record(versions={'InfiniOps':['old']})]);old=self.search()[0]['citation_id']
        with self.assertRaises(KeyError):self.store.read(old,'alice',['backend'],{'platform':'nvidia'})
        with self.assertRaises(KeyError):self.store.read(old,'alice',['backend'],{'versions':{'InfiniOps':'new'}})
        self.store.upsert([record(versions={'InfiniOps':['new']},revision='new')])
        with self.assertRaises(KeyError):self.store.read(old,'alice',['backend'],{'versions':{'InfiniOps':'new'}})
        self.assertFalse(self.store.read(old,'alice',['backend'])['is_current'])


if __name__=='__main__':unittest.main()
