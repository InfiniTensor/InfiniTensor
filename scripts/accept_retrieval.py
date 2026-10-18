#!/usr/bin/env python3
"""Offline reference, quality and process-isolated performance checks.

All measured inference engines are explicit. InfiniTensor never falls back to
the reference model. Performance numbers describe the recorded shared host.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from accept_m1 import (device_env, resource_sample, sha256, source_evidence,
                       sync_device, torch_device, write_json)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def reference(args, embedding=False):
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(1)
    path = args.embedding_model if embedding else args.model
    target = torch_device(args.device, args.device_id)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    factory = AutoModel if embedding else AutoModelForSequenceClassification
    model = factory.from_pretrained(path, local_files_only=True,
        attn_implementation='eager').eval().to(target)
    return model, tokenizer, target


def inputs(tokenizer, pairs):
    from pyinfinitensor.reranking import position_ids
    encoded = tokenizer([x[0] for x in pairs], [x[1] for x in pairs], padding=True,
        truncation=True, max_length=512, return_tensors='np', return_token_type_ids=False)
    result = {name: np.asarray(encoded[name], dtype=np.int64) for name in ('input_ids', 'attention_mask')}
    result['position_ids'] = position_ids(result['input_ids'], tokenizer.pad_token_id)
    return result


def forward(model, target, values, embedding=False):
    import torch
    from pyinfinitensor.embedding import fp32_inference_policy
    tensors = {name: torch.from_numpy(value).to(target) for name, value in values.items()}
    with fp32_inference_policy(), torch.inference_mode():
        result = model(**tensors)
        result = result.last_hidden_state[:, 0] if embedding else result.logits.reshape(-1)
    result = result.float().cpu().numpy()
    if embedding:
        result /= np.linalg.norm(result, axis=1, keepdims=True)
    return result


def session(args, optimized=True, mode=None):
    from pyinfinitensor.reranking import RerankerSession
    return RerankerSession(args.model, args.device, args.device_id,
        max_batch_size=8, execution_mode=mode or args.mode,
        memory_views=optimized, fuse_linear_bias=optimized)


def accuracy(args, output):
    from pyinfinitensor.embedding import EmbeddingSession, fp32_inference_policy
    engine = session(args, mode='graph')
    model, tokenizer, target = reference(args)
    rng = np.random.default_rng(20261018)
    shapes = [(1,4),(7,83),(3,511),(8,128),(1,4)]
    shapes += [(b,s) for b in (1,3,8) for s in (4,17,64,127,255,511,512)]
    shapes += [(int(rng.integers(1,9)),int(rng.integers(4,513))) for _ in range(100)]
    weights = engine.stub.weight_addresses.copy()
    uploads = (engine.stub.weight_upload_count, engine.stub.weight_upload_bytes)
    cases, previous = [], None
    progress = output/'progress.jsonl'
    with progress.open('w', encoding='utf-8') as log, fp32_inference_policy():
        for index, (batch, length) in enumerate(shapes):
            ids = rng.integers(4, 30000, (batch,length), dtype=np.int64)
            values = {'input_ids':ids,'attention_mask':np.ones_like(ids),
                      'position_ids':np.broadcast_to(np.arange(2,length+2,dtype=np.int64),(batch,length)).copy()}
            expected = forward(model,target,values)
            actual = engine.stub.run_numpy(values)['scores']
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3)
            # Same shape, changed contents must exercise replay, not stale input.
            values['input_ids'] = rng.integers(4,30000,(batch,length),dtype=np.int64)
            expected2 = forward(model,target,values)
            actual2 = engine.stub.run_numpy(values)['scores']
            np.testing.assert_allclose(actual2, expected2, atol=1e-4, rtol=1e-3)
            if previous is not None:
                np.testing.assert_array_equal(*previous)
            previous = actual2, actual2.copy()
            assert weights == engine.stub.weight_addresses
            assert uploads == (engine.stub.weight_upload_count, engine.stub.weight_upload_bytes)
            record = {'index':index,'shape':[batch,length],
                      'max_abs_error':float(max(np.max(abs(actual-expected)),np.max(abs(actual2-expected2))))}
            cases.append(record); log.write(json.dumps(record)+'\n'); log.flush()
    pairs = [('', ''),('知识库是什么？','向量数据库能够存储文档语义。'),
             ('Python 如何安装？','安装 Python 后可以运行程序。'),('检索','长文本。'*700)]
    pairs = (pairs*10)[:37]
    expected = np.concatenate([forward(model,target,inputs(tokenizer,pairs[start:start+8]))
                               for start in range(0,len(pairs),8)])
    actual = engine.score(pairs, batching='length')
    np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=1e-3)
    np.testing.assert_allclose(engine.score([pairs[1]]),actual[1:2],atol=1e-4,rtol=1e-3)
    assert engine.score([]).shape == (0,)
    assert engine.execution_stats()['replays'] > 0
    result = {'status':'PASS','dynamic_cases':cases,'changed_input_replays':len(cases),
        'text_pairs':len(pairs),'text_max_abs_error':float(abs(actual-expected).max()),
        'weights_resident':True,'uploads':uploads,'execution':engine.execution_stats(),
        'capabilities':engine.stub.check_capabilities(), 'memory':engine.stub.memory_stats()}
    write_json(output/'reranker.json',result)
    del engine,model; gc.collect()
    # The shared allocator/MatMul changes also affect the embedding model.
    embedding = EmbeddingSession(args.embedding_model,args.device,args.device_id,
        max_batch_size=8,memory_views=True,fuse_linear_bias=True,execution_mode='graph')
    model,tokenizer,target = reference(args,embedding=True)
    checks = []
    for batch,length in [(1,1),(7,83),(3,511),(8,128),(1,1)]:
        ids=rng.integers(104,20000,(batch,length),dtype=np.int64)
        values={'input_ids':ids,'attention_mask':np.ones_like(ids),'token_type_ids':np.zeros_like(ids)}
        with fp32_inference_policy():
            actual=embedding.stub.run_numpy(values)['embeddings']
        actual=actual/np.linalg.norm(actual,axis=1,keepdims=True)
        expected=forward(model,target,values,embedding=True)
        np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=1e-3)
        cosine=np.sum(actual*expected,axis=1)
        assert np.min(cosine)>=0.99999
        checks.append({'shape':[batch,length],'min_cosine':float(cosine.min())})
    return {'status':'PASS','reranker':result,'embedding_regression':checks}


def metrics(order, positives):
    relevant=set(positives)
    hits=[int(identity in relevant) for identity in order[:10]]
    ideal=sum(1/np.log2(i+2) for i in range(min(10,len(relevant))))
    dcg=sum(hit/np.log2(i+2) for i,hit in enumerate(hits))
    return {'ndcg10':float(dcg/ideal),'mrr10':next((1/(i+1) for i,h in enumerate(hits) if h),0.),
            'recall10':sum(hits)/len(relevant),'recall50':len(set(order[:50])&relevant)/len(relevant)}


def quality(args,output):
    from pyinfinitensor.embedding import EmbeddingSession, BGE_QUERY_PREFIX, cosine_top_k
    from pyinfinitensor.retrieval import KnowledgeIndex
    documents,queries=rows(Path(args.dataset)/'documents.jsonl'),rows(Path(args.dataset)/'queries.jsonl')
    embed=EmbeddingSession(args.embedding_model,args.device,args.device_id,max_batch_size=8,
        memory_views=True,fuse_linear_bias=True)
    index=KnowledgeIndex.build(documents,embed,batch_size=8);index.save(output/'index.npz')
    query_vectors=embed.encode([q['query'] for q in queries],kind='query',batch_size=8)
    # The optimized recall stage is checked against independent same-device eager forward.
    model,tok,target=reference(args,embedding=True)
    def encode(texts):
        result=[]
        for start in range(0,len(texts),8):
            values=tok(texts[start:start+8],padding=True,truncation=True,max_length=512,return_tensors='np')
            result.append(forward(model,target,{k:np.asarray(v,np.int64) for k,v in values.items()},True))
        return np.concatenate(result)
    expected_docs=encode(index.texts);expected_queries=encode([BGE_QUERY_PREFIX+q['query'] for q in queries])
    np.testing.assert_allclose(index.vectors,expected_docs,atol=1e-4,rtol=1e-3)
    np.testing.assert_allclose(query_vectors,expected_queries,atol=1e-4,rtol=1e-3)
    assert min(np.sum(index.vectors*expected_docs,axis=1))>=0.99999
    del model,embed;gc.collect()
    reranker=session(args,mode='normal');model,tok,target=reference(args)
    records=[]
    performance_requests=[]
    with (output/'queries.jsonl').open('w',encoding='utf-8') as log:
        for q,vector in zip(queries,query_vectors):
            candidates=cosine_top_k(vector,index.vectors,min(50,len(documents)),index.ids)
            offsets=[x[0] for x in candidates];ids=[index.ids[i] for i in offsets]
            pairs=[(q['query'],index.texts[i]) for i in offsets]
            if len(performance_requests)<8:
                performance_requests.append({'query_id':q['id'],'pairs':pairs,'document_ids':ids})
            scores=reranker.score(pairs)
            expected=np.concatenate([forward(model,target,inputs(tok,pairs[i:i+8])) for i in range(0,len(pairs),8)])
            np.testing.assert_allclose(scores,expected,atol=1e-4,rtol=1e-3)
            order=np.lexsort((np.asarray(ids),-scores));reference_order=np.lexsort((np.asarray(ids),-expected))
            ranked=[ids[i] for i in order];ref_ranked=[ids[i] for i in reference_order]
            overlap=len(set(ranked[:10])&set(ref_ranked[:10]))/10
            record={'id':q['id'],'recall':metrics(ids,q['positive_ids']),
                'rerank':metrics(ranked,q['positive_ids']),'top10_overlap':overlap,
                'max_abs_error':float(abs(scores-expected).max()),'recall_ids':ids,
                'reranked_ids':ranked,'scores':scores.tolist(),'reference_scores':expected.tolist()}
            records.append(record);log.write(json.dumps(record)+'\n');log.flush()
    summary={key:{metric:float(np.mean([r[key][metric] for r in records]))
                  for metric in records[0][key]} for key in ('recall','rerank')}
    overlap=float(np.mean([r['top10_overlap'] for r in records]));assert overlap>=0.99
    write_json(output/'performance-requests.json',performance_requests)
    return {'status':'PASS','queries':len(queries),'documents':len(documents),'metrics':summary,
            'mean_top10_reference_overlap':overlap,'scope':'fixed application subset, not a full benchmark',
            'dataset_manifest':json.loads((Path(args.dataset)/'manifest.json').read_text())}


def benchmark_worker(args,output):
    from pyinfinitensor.embedding import fp32_inference_policy
    from pyinfinitensor.batching import plan_batches
    from pyinfinitensor.reranking import position_ids
    import torch
    from transformers import AutoTokenizer
    started=time.perf_counter();optimized=args.variant!='baseline'
    batching='length' if args.variant in ('packed','graph','torch-packed') else 'input'
    engine=None;model=None
    if args.variant.startswith('torch'):
        model,tokenizer,target=reference(args)
    else:
        engine=session(args,optimized,mode='auto' if args.variant=='graph' else 'normal')
        tokenizer=engine.tokenizer
    loaded=time.perf_counter()
    docs=rows(Path(args.dataset)/'documents.jsonl')
    # Freeze an alternating short/long request before seeing engine timings.
    query=rows(Path(args.dataset)/'queries.jsonl')[0]['query']
    texts=sorted(docs,key=lambda d:(len(d['text']),d['id']))
    selected=[]
    for i in range(32):
        selected.extend([texts[i],texts[-1-i]])
    pairs=[(query,d['text']) for d in selected]
    request_file=Path(args.requests) if args.requests else None
    real_requests=json.loads(request_file.read_text()) if request_file else []
    fixed=[(1,32),(1,128),(1,512),(8,128),(8,512)]
    cases=[]
    def torch_score(items):
        encoded=tokenizer([x[0] for x in items],[x[1] for x in items],padding=False,truncation=True,
            max_length=512,return_token_type_ids=False)
        lengths=[len(x) for x in encoded['input_ids']]
        batches=plan_batches(lengths,8,4096,batching)
        scores=np.empty(len(items),np.float32);padding=0
        for indices in batches:
            length=max(lengths[i] for i in indices);shape=(len(indices),length)
            ids=np.full(shape,tokenizer.pad_token_id,np.int64);mask=np.zeros_like(ids)
            for row,index in enumerate(indices):
                ids[row,:lengths[index]]=encoded['input_ids'][index];mask[row,:lengths[index]]=1
            values={'input_ids':ids,'attention_mask':mask,'position_ids':position_ids(ids)}
            scores[indices]=forward(model,target,values);padding+=ids.size
        return scores,{'effective_tokens':sum(lengths),'padded_tokens':padding}
    with fp32_inference_policy():
        for name in [f'{b}x{s}' for b,s in fixed]+['mixed64']+(['retrieval50'] if real_requests else []):
            request_counter=0
            if name not in ('mixed64','retrieval50'):
                batch,length=map(int,name.split('x'));rng=np.random.default_rng(818)
                ids=rng.integers(4,30000,(batch,length),dtype=np.int64)
                values={'input_ids':ids,'attention_mask':np.ones_like(ids),'position_ids':position_ids(ids)}
                def once():
                    if engine:
                        result=engine.stub.run_numpy(values)['scores'];timing=dict(engine.stub.last_timing)
                    else:
                        result=forward(model,target,values);timing={}
                    return result,dict(timing,effective_tokens=ids.size,padded_tokens=ids.size)
            else:
                def once():
                    nonlocal request_counter
                    items=real_requests[request_counter%len(real_requests)]['pairs'] if name=='retrieval50' else pairs
                    request_counter+=1
                    if engine:
                        result=engine.score(items,batching=batching);timing=dict(engine.last_timing)
                        return result,timing
                    return torch_score(items)
            sync_device(args.device);t=time.perf_counter();once();sync_device(args.device)
            first=time.perf_counter()-t
            for _ in range(args.warmup):once()
            samples=[];detail=[];before=resource_sample(args.device,engine)
            for _ in range(args.runs):
                sync_device(args.device);t=time.perf_counter();result,timing=once();sync_device(args.device)
                samples.append(time.perf_counter()-t);detail.append(timing)
                assert np.isfinite(result).all()
            item={'case':name,'first_request_s':first,'samples_s':samples,'detail':detail,
                'p50_ms':float(np.percentile(samples,50)*1000),'p95_ms':float(np.percentile(samples,95)*1000),
                'pairs_per_s':len(result)/np.mean(samples),'effective_tokens_per_s':np.mean([d['effective_tokens'] for d in detail])/np.mean(samples),
                'padded_tokens_per_s':np.mean([d['padded_tokens'] for d in detail])/np.mean(samples),
                'resources_before':before,'resources_after':resource_sample(args.device,engine)}
            if engine:item.update(execution=engine.execution_stats(),memory=engine.stub.memory_stats())
            cases.append(item);write_json(output/'progress.json',{'completed':cases})
    return {'status':'PASS','variant':args.variant,'load_s':loaded-started,'cases':cases,
        'warmup':args.warmup,'runs':args.runs,'shared_environment':True,
        'timing_boundary':'synchronized host submission, forward and output transfer; mixed64 also includes tokenizer/packing',
        'mixed_document_ids':[d['id'] for d in selected],
        'mixed64_scope':'deterministic alternating shortest/longest texts; padding stress case',
        'retrieval_requests_sha256':sha256(request_file) if request_file else None}


def benchmark(args,output):
    variants=['baseline','engine','packed','graph','torch-input','torch-packed']
    records=[]
    for repeat in range(3):
        order=variants[repeat*2:]+variants[:repeat*2]
        for variant in order:
            destination=output/f'round-{repeat+1}'/variant
            command=[sys.executable,__file__,'--stage','worker','--model',args.model,
                '--embedding-model',args.embedding_model,'--dataset',args.dataset,'--device',args.device,
                '--device-id',str(args.device_id),'--output',str(destination),'--variant',variant,
                '--warmup',str(args.warmup),'--runs',str(args.runs)]
            if args.requests:command+=['--requests',args.requests]
            destination.mkdir(parents=True,exist_ok=True)
            with (destination/'process.log').open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            record={'round':repeat+1,'variant':variant,'report':str(destination/'result.json')}
            records.append(record);write_json(output/'progress.json',{'completed':records})
    return {'status':'PASS','processes':records,'rotated_order':True,'shared_environment':True}


def stress(args,output):
    from pyinfinitensor.embedding import fp32_inference_policy
    engine=session(args,mode='graph');model,tok,target=reference(args)
    rng=np.random.default_rng(818)
    shapes=[(1,4),(8,128),(3,511),(8,512),(1,17)]
    def values(shape):
        from pyinfinitensor.reranking import position_ids
        ids=rng.integers(4,30000,shape,dtype=np.int64)
        return {'input_ids':ids,'attention_mask':np.ones_like(ids),'position_ids':position_ids(ids)}
    with fp32_inference_policy():
        for shape in shapes:
            for _ in range(3):engine.stub.run_numpy(values(shape))
        weights=engine.stub.weight_addresses.copy()
        uploads=(engine.stub.weight_upload_count,engine.stub.weight_upload_bytes)
        samples=[];previous=None;started=time.perf_counter()
        with (output/'progress.jsonl').open('w') as log:
            for index in range(args.requests_count):
                # Hold shapes for two requests, so every second request replays.
                data=values(shapes[(index//2)%len(shapes)])
                actual=engine.stub.run_numpy(data)['scores']
                assert np.isfinite(actual).all()
                if previous is not None:np.testing.assert_array_equal(*previous)
                previous=actual,actual.copy()
                assert weights==engine.stub.weight_addresses
                assert uploads==(engine.stub.weight_upload_count,engine.stub.weight_upload_bytes)
                if index%100==0 or index==args.requests_count-1:
                    np.testing.assert_allclose(actual,forward(model,target,data),atol=1e-4,rtol=1e-3)
                    sample=dict(resource_sample(args.device,engine),request=index+1,elapsed_s=time.perf_counter()-started)
                    samples.append(sample);log.write(json.dumps(sample)+'\n');log.flush()
    assert engine.execution_stats()['replays']>0
    return {'status':'PASS','requests':args.requests_count,'elapsed_s':time.perf_counter()-started,
        'samples':samples,'weights_resident':True,'execution':engine.execution_stats(),
        'scope':'new-path dynamic stress regression; does not replace six-endpoint M1 two-hour certification'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['accuracy','quality','benchmark','worker','stress'],required=True)
    parser.add_argument('--model',required=True);parser.add_argument('--embedding-model',required=True)
    parser.add_argument('--dataset',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--requests',help='frozen first eight recall requests generated by quality stage')
    parser.add_argument('--device',choices=['nvidia','cambricon','ascend'],default='nvidia')
    parser.add_argument('--device-id',type=int,default=0);parser.add_argument('--mode',default='normal')
    parser.add_argument('--variant',default='engine');parser.add_argument('--warmup',type=int,default=20)
    parser.add_argument('--runs',type=int,default=100)
    parser.add_argument('--requests-count',type=int,default=1000)
    args=parser.parse_args()
    if args.runs<1 or args.warmup<0 or args.requests_count<1:
        parser.error('runs/requests-count must be positive and warmup nonnegative')
    # Keep stress duration explicit in the report; there is no idle-time pass.
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    envelope={'stage':args.stage,'environment':device_env(args.device),'source':source_evidence(),
              'model_sha256':sha256(Path(args.model)/'model.onnx'),
              'dataset_manifest':json.loads((Path(args.dataset)/'manifest.json').read_text()),
              'started_unix_s':time.time()}
    try:
        result={'accuracy':accuracy,'quality':quality,'benchmark':benchmark,'worker':benchmark_worker,'stress':stress}[args.stage](args,output)
        envelope.update(status='PASS',result=result)
    except BaseException as error:
        envelope.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
        write_json(output/'result.json',envelope);raise
    envelope['ended_unix_s']=time.time();write_json(output/'result.json',envelope)
    print(json.dumps({'status':'PASS','stage':args.stage,'output':str(output)}),flush=True)


if __name__=='__main__':main()
