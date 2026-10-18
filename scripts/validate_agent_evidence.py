#!/usr/bin/env python3
"""Real-query agent retrieval pilot; no generated questions or LLM fallback."""
import argparse
import asyncio
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from accept_m1 import device_env,resource_sample,sync_device,write_json
from infinitensor_evidence import EvidenceStore
from infinitensor_evidence.engine import InfiniTensorEngine
from infinitensor_evidence.service import EvidenceService


def rows(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def signature(result):return result['context'],[c['citation_id'] for c in result['citations']]


async def protocol(args,store_path,query,output):
    from mcp import ClientSession,StdioServerParameters
    from mcp.client.stdio import stdio_client
    parameters=StdioServerParameters(command=sys.executable,args=['-m','infinitensor_evidence','serve',
        '--store',str(store_path),'--principal','validation-user','--namespace','infinitensor',
        '--embedding-model',args.embedding_model,'--reranker-model',args.reranker_model,
        '--device',args.device,'--device-id',str(args.device_id)],env=dict(os.environ))
    started=time.perf_counter()
    with (output/'mcp-server.log').open('w') as log:
        async with stdio_client(parameters,errlog=log) as (read,write):
            async with ClientSession(read,write) as session:
                await session.initialize();initialized=time.perf_counter()
                tools=(await session.list_tools()).tools
                assert {x.name for x in tools}=={'search_evidence','read_evidence'}
                first=await session.call_tool('search_evidence',{'query':query,'recall_k':24})
                assert not first.isError and first.structuredContent['citations']
                citation=first.structuredContent['citations'][0]['citation_id']
                passage=await session.call_tool('read_evidence',{'citation_id':citation})
                assert not passage.isError and passage.structuredContent['is_current']
                repeated=await asyncio.gather(*[session.call_tool('search_evidence',{'query':query,'recall_k':24}) for _ in range(3)])
                assert all(not x.isError and signature(x.structuredContent)==signature(first.structuredContent) for x in repeated)
                invalid=await session.call_tool('search_evidence',{'query':query,'top_k':True})
                assert invalid.isError
    return {'status':'PASS','startup_s':initialized-started,'tools':[x.name for x in tools],
            'concurrent_requests':3,'stable_citations':True,'strict_input_validation':True}


def reference_check(args,engine,pairs):
    import torch
    from transformers import AutoModel,AutoModelForSequenceClassification
    from pyinfinitensor.embedding import fp32_inference_policy,BGE_QUERY_PREFIX
    target={'nvidia':'cuda','cambricon':'mlu','ascend':'npu'}[args.device]+':'+str(args.device_id)
    result={}
    with fp32_inference_policy(),torch.inference_mode():
        model=AutoModel.from_pretrained(args.embedding_model,local_files_only=True,attn_implementation='eager').eval().to(target)
        query=pairs[0][0]
        values=engine.embedding.tokenizer([BGE_QUERY_PREFIX+query],return_tensors='pt',padding=True,truncation=False)
        expected=model(**{k:v.to(target) for k,v in values.items()}).last_hidden_state[:,0].float().cpu().numpy()[0]
        expected/=np.linalg.norm(expected);actual=engine.encode_query(query)
        np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=1e-3)
        result['embedding_max_abs_error']=float(abs(actual-expected).max())
        del model;gc.collect()
        model=AutoModelForSequenceClassification.from_pretrained(args.reranker_model,local_files_only=True,attn_implementation='eager').eval().to(target)
        values=engine.reranker.tokenizer([x[0] for x in pairs],[x[1] for x in pairs],padding=True,
            truncation=False,return_tensors='pt',return_token_type_ids=False)
        expected=model(**{k:v.to(target) for k,v in values.items()}).logits.reshape(-1).float().cpu().numpy()
        actual=engine.score(pairs)
        np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=1e-3)
        result['reranker_max_abs_error']=float(abs(actual-expected).max())
        del model;gc.collect()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--embedding-model',required=True)
    p.add_argument('--reranker-model',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',choices=['nvidia','cambricon','ascend'],default='nvidia');p.add_argument('--device-id',type=int,default=0)
    p.add_argument('--runs',type=int,default=84);p.add_argument('--warmup',type=int,default=5)
    args=p.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    report={'status':'RUNNING','environment':device_env(args.device),'shared_environment':True}
    try:
        started=time.perf_counter()
        documents=rows(Path(args.data)/'documents.jsonl');queries=rows(Path(args.data)/'queries.jsonl')
        by_id={d['id']:d for d in documents}
        engine=InfiniTensorEngine(args.embedding_model,args.reranker_model,args.device,args.device_id,cache=False)
        model_loaded=time.perf_counter();store=EvidenceStore(output/'evidence.db')
        build=store.upsert(documents,engine.encode_documents,engine.fingerprint,engine.chunks)
        built=time.perf_counter();service=EvidenceService(store,'validation-user',['infinitensor'],engine)
        residency=[engine.embedding.stub.weight_addresses.copy(),engine.reranker.stub.weight_addresses.copy()]
        uploads=[engine.embedding.stub.weight_upload_bytes,engine.reranker.stub.weight_upload_bytes]
        counters_before=engine.metrics();no_op=store.upsert(documents,engine.encode_documents,engine.fingerprint,engine.chunks)
        assert no_op['updated_documents']==0 and counters_before==engine.metrics()
        source_rows=store.eligible_chunks('validation-user',['infinitensor'])
        pairs=[(queries[0]['query'],row['embedding_text']) for row in source_rows[:2]]
        precision=reference_check(args,engine,pairs)
        quality={};saved={};traces=[]
        variants=[('lexical','lexical',False),('vector','vector',False),('hybrid','hybrid',False),('reranked','hybrid',True)]
        with (output/'retrieval-traces.jsonl').open('w',encoding='utf-8') as log:
            for name,mode,rerank in variants:
                records=[]
                for query in queries:
                    result=service.search_evidence(query['query'],top_k=5,recall_k=24,mode=mode,rerank=rerank)
                    assert result['budget']['used']<=result['budget']['limit']
                    for citation in result['citations']:
                        read=store.read(citation['citation_id'],'validation-user',['infinitensor'])
                        doc=by_id[citation['document_id']]
                        assert read['body']==doc['text'][citation['start']:citation['end']]
                        assert read['source_revision']==doc['revision']
                    paths=[c['document_id'].split(':L')[0] for c in result['citations']]
                    gold=set(query['relevant_paths']);rank=next((i+1 for i,path in enumerate(paths) if path in gold),None)
                    record={'query_id':query['id'],'variant':name,'source_uri':query['source_uri'],
                        'has_guidance_label':bool(gold),'guidance_hit5':bool(rank),'reciprocal_rank5':1/rank if rank else 0.,
                        'result':result}
                    records.append(record);log.write(json.dumps(record,ensure_ascii=False)+'\n');log.flush()
                    saved[name,query['id']]=signature(result)
                labeled=[r for r in records if r['has_guidance_label']]
                quality[name]={'guidance_hit5':float(np.mean([r['guidance_hit5'] for r in labeled])),
                    'guidance_mrr5':float(np.mean([r['reciprocal_rank5'] for r in labeled])),'labeled_queries':len(labeled)}
        write_json(output/'quality.json',{'status':'PASS','metrics':quality,'scope':'curated related-document guidance, not historical bug resolution or LLM task success'})
        benchmark=[]
        for repeat in range(3):
            order=['lexical','hybrid_default','uncached','exact_cached']
            order=order[repeat:]+order[:repeat]
            for variant in order:
                cached=variant=='exact_cached';engine.cache.clear()
                mode='lexical' if variant=='lexical' else 'hybrid'
                rerank=variant in {'uncached','exact_cached'}
                expected='lexical' if variant=='lexical' else ('reranked' if rerank else 'hybrid')
                engine.cache.query_capacity=128 if cached else 0;engine.cache.pair_capacity=4096 if cached else 0
                if cached:
                    for q in queries:service.search_evidence(q['query'],recall_k=24,rerank=True)
                for i in range(args.warmup):
                    service.search_evidence(queries[i%len(queries)]['query'],recall_k=24,mode=mode,rerank=rerank)
                before=engine.metrics();samples=[];resources=resource_sample(args.device)
                for i in range(args.runs):
                    query=queries[i%len(queries)];sync_device(args.device);t=time.perf_counter()
                    result=service.search_evidence(query['query'],recall_k=24,mode=mode,rerank=rerank)
                    sync_device(args.device);elapsed=time.perf_counter()-t
                    assert signature(result)==saved[expected,query['id']]
                    samples.append({'query_id':query['id'],'elapsed_s':elapsed,'timing':result['timing'],
                                    'context_tokens':result['budget']['used'],'citations':len(result['citations'])})
                after=engine.metrics()
                record={'round':repeat+1,'variant':variant,'samples':samples,
                    'p50_ms':float(np.percentile([s['elapsed_s'] for s in samples],50)*1000),
                    'p95_ms':float(np.percentile([s['elapsed_s'] for s in samples],95)*1000),
                    'resources_before':resources,'resources_after':resource_sample(args.device),
                    'engine_before':before,'engine_after':after}
                if cached:
                    assert before['embedding_execution']==after['embedding_execution']
                    assert before['reranker_execution']==after['reranker_execution']
                benchmark.append(record);write_json(output/'benchmark-progress.json',{'completed':benchmark})
        # A separate contract store exercises real model updates without
        # contaminating the public-query quality corpus or performance samples.
        lifecycle=EvidenceStore(output/'update-contract.db')
        item=dict(documents[0]);item['readers']=['validation-user']
        lifecycle.upsert([item],engine.encode_documents,engine.fingerprint,engine.chunks)
        old=lifecycle.eligible_chunks('validation-user',['infinitensor'])[0]['citation_id']
        item.update(revision=item['revision']+'-contract-update',text=item['text']+'\nUpdated evidence contract.')
        lifecycle.upsert([item],engine.encode_documents,engine.fingerprint,engine.chunks)
        assert not lifecycle.read(old,'validation-user',['infinitensor'])['is_current']
        lifecycle.withdraw('infinitensor',item['id'])
        try:lifecycle.read(old,'validation-user',['infinitensor'])
        except KeyError:pass
        else:raise AssertionError('withdrawn citation remained readable')
        lifecycle.close()
        before=engine.metrics()
        try:engine.encode_query('超长查询 '*300)
        except ValueError:pass
        else:raise AssertionError('query limit was not checked')
        assert before==engine.metrics()
        for stub,w,b in zip([engine.embedding.stub,engine.reranker.stub],residency,uploads):
            assert stub.weight_addresses==w and stub.weight_upload_bytes==b
        report.update(status='PASS',precision=precision,build=build,no_op_update=no_op,
            startup={'model_load_s':model_loaded-started,'index_s':built-model_loaded},quality=quality,
            dataset=json.loads((Path(args.data)/'manifest.json').read_text()),benchmark=benchmark,
            weights_resident=True,update_contract='PASS',length_guard='PASS',
            timing_scope='one resident process, three rotated rounds; synchronized complete retrieval request',
            cache_scope='exact repeated queries and exact query/passage pairs after priming all 42 real queries; not semantic answer caching',
            model_metrics=engine.metrics())
        store.close();del service,engine;gc.collect()
        report['mcp']=asyncio.run(protocol(args,output/'evidence.db',queries[0]['query'],output))
    except BaseException as error:
        import traceback
        report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
        write_json(output/'result.json',report);raise
    write_json(output/'result.json',report)
    print(json.dumps({'status':'PASS','output':str(output),'quality':quality}),flush=True)


if __name__=='__main__':main()
