"""InfiniTensor-only models with bounded exact-input reuse for agent loops."""
from collections import OrderedDict
import hashlib
import json
import threading

import numpy as np

from .text import positive


class InferenceCache:
    def __init__(self,query_encoder,pair_scorer,query_capacity=128,pair_capacity=4096):
        for value in (query_capacity,pair_capacity):
            if isinstance(value,bool) or not isinstance(value,int) or not 0<=value<=100000:raise ValueError('invalid cache capacity')
        self.query_encoder,self.pair_scorer=query_encoder,pair_scorer
        self.query_capacity,self.pair_capacity=query_capacity,pair_capacity
        self.queries,self.pairs=OrderedDict(),OrderedDict();self.lock=threading.RLock()
        self.stats={'query_hits':0,'query_misses':0,'pair_hits':0,'pair_misses':0}

    @staticmethod
    def _put(cache,key,value,capacity):
        if not capacity:return
        cache[key]=value;cache.move_to_end(key)
        while len(cache)>capacity:cache.popitem(last=False)

    def query(self,text):
        with self.lock:
            if text in self.queries:
                self.stats['query_hits']+=1;self.queries.move_to_end(text)
                return self.queries[text].copy()
            value=np.array(self.query_encoder(text),dtype=np.float32,copy=True).reshape(-1)
            if not np.isfinite(value).all() or not np.isclose(np.linalg.norm(value),1.,atol=1e-5):raise ValueError('invalid query embedding')
            self.stats['query_misses']+=1
            self._put(self.queries,text,value,self.query_capacity)
            return value.copy()

    def score(self,pairs):
        with self.lock:
            pairs=list(pairs);keys=[hashlib.sha256(json.dumps(pair,ensure_ascii=False).encode()).hexdigest() for pair in pairs]
            missing={}
            for key,pair in zip(keys,pairs):
                if key in self.pairs:
                    self.stats['pair_hits']+=1;self.pairs.move_to_end(key)
                else:missing.setdefault(key,pair)
            fresh={}
            if missing:
                values=np.asarray(self.pair_scorer(list(missing.values())),dtype=np.float32)
                if values.shape!=(len(missing),) or not np.isfinite(values).all():raise ValueError('invalid reranker output')
                self.stats['pair_misses']+=len(missing)
                fresh=dict(zip(missing,values))
                # Build output before eviction; a request can exceed capacity.
            result=np.asarray([fresh[key] if key in fresh else self.pairs[key] for key in keys],dtype=np.float32)
            for key,value in fresh.items():self._put(self.pairs,key,float(value),self.pair_capacity)
            return result

    def clear(self):
        with self.lock:self.queries.clear();self.pairs.clear()


class InfiniTensorEngine:
    def __init__(self,embedding_model,reranker_model,device='nvidia',device_id=0,batch_size=8,
                 cache=True,execution_mode='normal'):
        from pyinfinitensor.embedding import EmbeddingSession
        from pyinfinitensor.reranking import RerankerSession
        positive(batch_size,'batch_size',8)
        self.embedding=EmbeddingSession(embedding_model,device,device_id,max_batch_size=batch_size,
            memory_views=True,fuse_linear_bias=True,execution_mode=execution_mode)
        self.reranker=RerankerSession(reranker_model,device,device_id,max_batch_size=batch_size,
            execution_mode=execution_mode)
        self.batch_size=batch_size
        self.fingerprint=hashlib.sha256(json.dumps({'package':self.embedding.package_identity['fingerprint'],
            'preprocessing':'BGE Chinese query instruction; CLS; L2; right padding; 512',
            'document_format':'title-newline-source-span-v1'},sort_keys=True).encode()).hexdigest()
        self.tokenizer_name='BAAI/bge-reranker-base offline tokenizer (no special tokens)'
        self.cache=InferenceCache(lambda text:self.embedding.encode([text],kind='query',batch_size=1)[0],
            lambda pairs:self.reranker.score(pairs),128 if cache else 0,4096 if cache else 0)
        self.lock=threading.RLock()
        self._token_counts=OrderedDict()

    def _count(self,tokenizer,text):
        key=(id(tokenizer),hashlib.sha256(text.encode('utf-8')).digest())
        with self.lock:
            if key in self._token_counts:
                self._token_counts.move_to_end(key);return self._token_counts[key]
            value=len(tokenizer(text,add_special_tokens=False,truncation=False,verbose=False)['input_ids'])
            InferenceCache._put(self._token_counts,key,value,2048)
            return value

    def count_tokens(self,text):return self._count(self.reranker.tokenizer,text)

    def _query(self,text):
        if not isinstance(text,str) or not text.strip():raise ValueError('query must be nonempty')
        if max(self._count(self.embedding.tokenizer,text),self.count_tokens(text))>96:
            raise ValueError('submit a focused search query of at most 96 tokens')

    def _passage(self,text):
        if max(self._count(self.embedding.tokenizer,text),self.count_tokens(text))>384:
            raise ValueError('passage exceeds the certified budget; rebuild with token-aware chunks')

    def encode_documents(self,texts):
        texts=list(texts)
        for text in texts:self._passage(text)
        with self.lock:return self.embedding.encode(texts,kind='passage',batch_size=self.batch_size)

    def encode_query(self,text):
        self._query(text)
        with self.lock:return self.cache.query(text)

    def score(self,pairs):
        pairs=list(pairs)
        for query,passage in pairs:self._query(query);self._passage(passage)
        with self.lock:return self.cache.score(pairs)

    def chunks(self,text,maximum=300,overlap=40):
        """Keep source offsets; fit BOTH tokenizers without silent truncation."""
        positive(maximum,'maximum',320)
        if not isinstance(overlap,int) or not 0<=overlap<maximum:raise ValueError('invalid overlap')
        offsets=self.reranker.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True,
            truncation=False,verbose=False)['offset_mapping']
        if not offsets:return []
        spans=[];start=0
        while start<len(offsets):
            end=min(start+maximum,len(offsets))
            begin_char=offsets[start][0] if start else 0
            while end>start:
                end_char=len(text) if end==len(offsets) else offsets[end-1][1]
                body=text[begin_char:end_char]
                if self._count(self.embedding.tokenizer,body)<=maximum:break
                end-=max(1,(end-start)//8)
            if end<=start:raise ValueError('a token cannot fit chunk budget')
            spans.append((begin_char,end_char))
            if end==len(offsets):break
            start=max(start+1,end-overlap)
        return spans

    def metrics(self):
        return {'cache':dict(self.cache.stats,query_entries=len(self.cache.queries),pair_entries=len(self.cache.pairs)),
                'embedding_execution':self.embedding.execution_stats(),'reranker_execution':self.reranker.execution_stats(),
                'embedding_timing':dict(self.embedding.last_timing),'reranker_timing':dict(self.reranker.last_timing)}
