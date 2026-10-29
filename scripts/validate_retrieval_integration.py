#!/usr/bin/env python3
"""Cross-device migration checks for embedding, reranking and agent evidence.

This is a correctness regression, not the M1 long-running acceptance or a
performance benchmark. Reference models are used only by this validation tool.
"""
import argparse
import asyncio
import gc
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np


def residency(stub):
    return (stub.weight_upload_count, stub.weight_upload_bytes, stub.weight_addresses.copy())


def provider_plan(stub, device):
    plan = stub.check_capabilities()
    assert plan['device_type'] == device, plan
    for step in plan['steps']:
        if step['provider_cpp_type'] == 'InfiniRT':
            continue
        name = subprocess.check_output(['c++filt', '-t', step['provider_cpp_type']], text=True).strip()
        match = re.search(r'Operator<.*\(infini::rt::Device::Type\)([0-9]+), ([0-9]+)ul>', name)
        assert match and int(match[1]) == plan['device_code'], name
        step['provider_demangled'] = name
    return plan


async def mcp_check(args, namespaces):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    command = ['-m','infinitensor_evidence','serve','--store',str(args.output/'evidence.db'),
               '--principal','validation-user','--embedding-model',args.embedding_model,
               '--reranker-model',args.reranker_model,'--device',args.device,
               '--device-id',str(args.device_id),'--rerank']
    for namespace in namespaces:
        command += ['--namespace',namespace]
    parameters = StdioServerParameters(command=sys.executable,args=command,env=dict(os.environ))
    with (args.output/'mcp-server.log').open('w',encoding='utf-8') as log:
        async with stdio_client(parameters,errlog=log) as (read,write):
            async with ClientSession(read,write) as session:
                await session.initialize()
                tools = {tool.name for tool in (await session.list_tools()).tools}
                assert tools == {'search_evidence','read_evidence'}
                reply = await session.call_tool('search_evidence',{'query':'动态输入与图捕获','recall_k':8,'top_k':3})
                assert not reply.isError and reply.structuredContent['citations']
                citation = reply.structuredContent['citations'][0]['citation_id']
                readback = await session.call_tool('read_evidence',{'citation_id':citation})
                assert not readback.isError and readback.structuredContent['is_current']
    return {'status':'PASS','tools':sorted(tools),'native_models':True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embedding-model', required=True)
    parser.add_argument('--reranker-model', required=True)
    parser.add_argument('--device', choices=['nvidia', 'cambricon', 'ascend'], required=True)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--random-shapes', type=int, default=100)
    parser.add_argument('--documents', type=Path, required=True, help='Existing evidence JSONL corpus')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.random_shapes < 0:
        parser.error('random-shapes must be nonnegative')
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'RUNNING', 'scope': 'correctness migration; no long-stability or performance claim',
              'device': args.device, 'device_id': args.device_id}

    def save():
        (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    try:
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification
        from infinitensor_evidence.engine import InfiniTensorEngine
        from infinitensor_evidence import EvidenceStore
        from infinitensor_evidence.service import EvidenceService
        from pyinfinitensor.embedding import fp32_inference_policy

        target = {'nvidia': 'cuda', 'cambricon': 'mlu', 'ascend': 'npu'}[args.device]
        if args.device == 'cambricon':
            import torch_mlu  # noqa: F401
        elif args.device == 'ascend':
            import torch_npu  # noqa: F401
        torch.manual_seed(315)
        getattr(torch, target).set_device(args.device_id)
        target += ':' + str(args.device_id)
        engine = InfiniTensorEngine(args.embedding_model, args.reranker_model, args.device, args.device_id)
        emb, rerank = engine.embedding, engine.reranker
        report['model_identity'] = {'embedding': emb.package_identity, 'reranker': rerank.package_identity}
        report['torch'] = torch.__version__
        states = [residency(emb.stub), residency(rerank.stub)]
        rng = np.random.default_rng(315)
        shapes = [(1,17), (7,83), (3,511), (16,128), (1,17)]
        shapes += [(b,s) for b in [1,3,8,17,32] for s in [2,17,64,127,255,511,512]]
        shapes += [(int(rng.integers(1,33)),int(rng.integers(1,513))) for _ in range(args.random_shapes)]
        shapes += [(1,1),(32,1),(1,512),(1,1)]
        report['embedding_cases'] = []
        with fp32_inference_policy(), torch.inference_mode():
            reference = AutoModel.from_pretrained(args.embedding_model, local_files_only=True,
                attn_implementation='eager').eval().to(target)
            old = saved = None
            for batch, sequence in shapes:
                values = {'input_ids': rng.integers(1,1000,(batch,sequence),dtype=np.int64),
                          'attention_mask': np.ones((batch,sequence),np.int64),
                          'token_type_ids': np.zeros((batch,sequence),np.int64)}
                expected = reference(**{k:torch.from_numpy(v).to(target) for k,v in values.items()}).last_hidden_state[:,0].cpu().numpy()
                raw = emb.stub.run_numpy(values)['embeddings']
                actual = raw / np.linalg.norm(raw,axis=1,keepdims=True)
                expected = expected / np.linalg.norm(expected,axis=1,keepdims=True)
                np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3)
                cosine = float(np.min(np.sum(actual*expected,axis=1)))
                assert cosine >= .99999 and residency(emb.stub) == states[0]
                if old is not None:
                    np.testing.assert_array_equal(old,saved)
                old, saved = raw, raw.copy()
                report['embedding_cases'].append({'shape':[batch,sequence], 'max_abs':float(abs(actual-expected).max()),'min_cosine':cosine})
                save()
            report['embedding_provider_plan'] = provider_plan(emb.stub,args.device)
            del reference
            gc.collect()
            reference = AutoModelForSequenceClassification.from_pretrained(args.reranker_model,
                local_files_only=True,attn_implementation='eager').eval().to(target)
            report['reranker_cases'] = []
            for pairs in [[('知识库检索','通过向量召回并精排文档'),('动态输入','一次加载后处理不同文本长度')],
                          [('', '')], [('长文本', '中文文档。'*700)],
                          [('推理引擎', '候选文档'+str(i)) for i in range(17)]]:
                tokens = rerank.tokenizer([p[0] for p in pairs],[p[1] for p in pairs],padding=True,
                    truncation=True,max_length=512,return_token_type_ids=False,return_tensors='pt')
                expected = reference(**{k:v.to(target) for k,v in tokens.items()}).logits.cpu().numpy().reshape(-1)
                actual = rerank.score(pairs)
                np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=1e-3)
                assert residency(rerank.stub) == states[1]
                report['reranker_cases'].append({'pairs':len(pairs),'shapes':rerank.last_timing['input_shapes'],
                    'max_abs':float(abs(actual-expected).max())})
                save()
            report['reranker_provider_plan'] = provider_plan(rerank.stub,args.device)
            del reference
            gc.collect()
        # Existing repository evidence is supplied by the caller, not generated
        # questions presented as proof of real-world retrieval quality.
        documents = [json.loads(line) for line in args.documents.read_text(encoding='utf-8').splitlines() if line.strip()]
        documents = documents[:8]
        namespaces = sorted({d['namespace'] for d in documents})
        store = EvidenceStore(args.output/'evidence.db')
        try:
            report['build'] = store.upsert(documents,engine.encode_documents,engine.fingerprint,engine.chunks)
            service = EvidenceService(store,'validation-user',namespaces,engine)
            result = service.search_evidence('动态输入与图捕获',top_k=3,recall_k=8,rerank=True)
            assert result['citations']
            for citation in result['citations']:
                read = store.read(citation['citation_id'],'validation-user',namespaces)
                assert read['is_current']
            repeated = service.search_evidence('动态输入与图捕获',top_k=3,recall_k=8,rerank=True)
            assert result['context'] == repeated['context']
            assert [c['citation_id'] for c in result['citations']] == [c['citation_id'] for c in repeated['citations']]
            report['evidence'] = {'citations':len(result['citations']),'stable_repeat':True,'metrics':engine.metrics()}
        finally:
            store.close()
        for stub,state in zip([emb.stub,rerank.stub],states):
            assert residency(stub) == state
        report['mcp'] = asyncio.run(mcp_check(args,namespaces))
        report['status'] = 'PASS'
    except Exception as error:
        report.update(status='FAIL',error=repr(error))
        raise
    finally:
        report['finished_unix'] = time.time()
        save()


if __name__ == '__main__':
    main()
