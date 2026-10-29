"""Real stdio client/server tests of the explicit CPU lexical baseline."""
import asyncio
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client

from infinitensor_evidence import EvidenceStore
from test_evidence_store import record


class TestEvidenceMCP(unittest.IsolatedAsyncioTestCase):
    def test_model_package_import_does_not_write_protocol_stdout(self):
        child = subprocess.run(
            [sys.executable, '-c', 'from pyinfinitensor.embedding import EmbeddingSession; '
             'from pyinfinitensor.reranking import RerankerSession'],
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(child.stdout, '')

    async def test_real_stdio_tools_preserve_host_identity_citations_and_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'evidence.db';store=EvidenceStore(path)
            store.upsert([record('allowed'),record('wrong-platform',platforms=['nvidia'],text='CUDA-ONLY 图捕获'),
                record('wrong-version',versions={'InfiniOps':['old']},text='OLD-VERSION 图捕获'),
                record('secret',readers=['bob'],text='SECRET-ONLY-BOB 图捕获')])
            secret=store.lexical_search('SECRET-ONLY-BOB','bob',['backend'])[0]['citation_id']
            outside_ids=[x['citation_id'] for x in store.eligible_chunks('alice',['backend']) if x['document_id'] in {'wrong-platform','wrong-version'}]
            store.close()
            params=StdioServerParameters(command=sys.executable,args=['-m','infinitensor_evidence','serve',
                '--store',str(path),'--principal','alice','--namespace','backend','--lexical-only','--platform','ascend','--version','InfiniOps=new'],
                env=dict(os.environ))
            async with stdio_client(params) as (read,write):
                async with ClientSession(read,write) as session:
                    await session.initialize()
                    tools=(await session.list_tools()).tools
                    self.assertEqual({tool.name for tool in tools},{'search_evidence','read_evidence'})
                    for tool in tools:
                        self.assertTrue(tool.annotations.readOnlyHint)
                        self.assertNotIn('principal',tool.inputSchema['properties'])
                        self.assertNotIn('namespace',tool.inputSchema['properties'])
                    result=await session.call_tool('search_evidence',{'query':'图捕获'})
                    self.assertFalse(result.isError);data=result.structuredContent
                    self.assertNotIn('SECRET-ONLY-BOB',data['context'])
                    identity=data['citations'][0]['citation_id']
                    reply=await session.call_tool('read_evidence',{'citation_id':identity})
                    self.assertEqual(reply.structuredContent['citation']['citation_id'],identity)
                    forbidden=await session.call_tool('read_evidence',{'citation_id':secret,'principal':'bob'})
                    self.assertTrue(forbidden.isError)
                    for identity in outside_ids:
                        denied=await session.call_tool('read_evidence',{'citation_id':identity})
                        self.assertTrue(denied.isError)
                    invalid=await session.call_tool('search_evidence',{'query':'图捕获','top_k':True})
                    self.assertTrue(invalid.isError)
                    outside=await session.call_tool('search_evidence',{'query':'图捕获','filters':{'platform':'nvidia'}})
                    self.assertTrue(outside.isError)
                    malformed=await session.call_tool('search_evidence',{'query':'图捕获','filters':{'versions':[]}})
                    self.assertTrue(malformed.isError)
                    concurrent=await asyncio.gather(*[session.call_tool('search_evidence',{'query':'图捕获'}) for _ in range(3)])
                    self.assertTrue(all(not reply.isError for reply in concurrent))


if __name__=='__main__':unittest.main()
