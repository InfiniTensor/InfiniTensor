"""Hybrid retrieval and budgeted citations over one explicitly scoped session."""
import re
import threading
import time

import numpy as np

from .store import sha,strings
from .text import positive


def render(row):
    return (f"[{row['citation_id']}] {row['title']}\n"
            f"Source: {row['source_uri']} @ {row['source_revision']} "
            f"chars {row['start']}:{row['end']}\n{row['body']}")


class EvidenceService:
    """Identity and namespace authority come from the host, never tool input.

    A lock owns all model access, including read tokenization, so two requests
    cannot mutate the same dynamic graph concurrently. Storage remains usable
    by independent indexing processes through SQLite transactions.
    """
    def __init__(self,store,principal,namespaces,engine=None,token_counter=None,
                 tokenizer_name=None,min_rerank_score=None):
        if not isinstance(principal,str) or not principal:raise ValueError('principal is required')
        self.store,self.principal,self.namespaces=store,principal,tuple(strings(namespaces,'namespaces'))
        self.engine=engine;self.lock=threading.RLock()
        self.count=token_counter or (engine.count_tokens if engine else len)
        self.tokenizer_name=tokenizer_name or (engine.tokenizer_name if engine else 'Unicode-code-points')
        if min_rerank_score is not None and not np.isfinite(min_rerank_score):raise ValueError('threshold must be finite')
        self.min_rerank_score=min_rerank_score

    def _vector(self,query,filters,limit):
        if self.engine is None:raise ValueError('vector retrieval requires an explicit InfiniTensor engine')
        if self.store.setting('embedding_fingerprint')!=self.engine.fingerprint:
            raise ValueError('index embedding package does not match this engine')
        # Filter BEFORE loading vectors or presenting text to either model.
        rows=self.store.eligible_chunks(self.principal,self.namespaces,filters)
        if not rows:return []
        if any(row['vector'] is None for row in rows):raise ValueError('index contains chunks without embeddings; rebuild the store')
        vector=np.asarray(self.engine.encode_query(query),dtype=np.float32).reshape(-1)
        if not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector),1.,atol=1e-5):raise ValueError('invalid query embedding')
        if any(row['dimension']!=vector.size for row in rows):raise ValueError('query embedding dimension differs from index')
        matrix=np.stack([np.frombuffer(row['vector'],dtype=np.float32) for row in rows])
        scores=matrix@vector
        order=np.lexsort((np.asarray([row['citation_id'] for row in rows]),-scores))[:limit]
        return [dict(rows[i],vector_score=float(scores[i])) for i in order]

    def search_evidence(self,query,filters=None,top_k=5,recall_k=50,context_budget=2048,
                        mode='hybrid',rerank=True):
        if not isinstance(query,str) or not query.strip():raise ValueError('query must be a nonempty string')
        positive(top_k,'top_k',20);positive(recall_k,'recall_k',200);positive(context_budget,'context_budget',32768)
        if top_k>recall_k:raise ValueError('top_k cannot exceed recall_k')
        if mode not in {'lexical','vector','hybrid'}:raise ValueError('unknown retrieval mode')
        if not isinstance(rerank,bool):raise ValueError('rerank must be boolean')
        # Validate even if no row happens to match, and before device work.
        self.store._scope(self.principal,self.namespaces,filters)
        with self.lock,self.store.snapshot():
            started=time.perf_counter()
            lexical=self.store.lexical_search(query,self.principal,self.namespaces,filters,recall_k) if mode!='vector' else []
            lexical_at=time.perf_counter()
            semantic=self._vector(query,filters,recall_k) if mode!='lexical' else []
            semantic_at=time.perf_counter()
            pool={};scores={}
            for stream in [lexical,semantic]:
                for rank,row in enumerate(stream,1):
                    identity=row['citation_id'];pool.setdefault(identity,{}).update(row)
                    scores[identity]=scores.get(identity,0.)+1/(60+rank)
            candidates=[dict(pool[key],rrf_score=scores[key]) for key in sorted(pool,key=lambda k:(-scores[k],k))[:recall_k]]
            ranked_at=time.perf_counter()
            if rerank and candidates:
                if self.engine is None:raise ValueError('reranking requires an explicit InfiniTensor engine')
                values=np.asarray(self.engine.score([(query,row['embedding_text']) for row in candidates]),dtype=np.float32)
                if values.shape!=(len(candidates),) or not np.isfinite(values).all():raise ValueError('invalid reranker output')
                for row,value in zip(candidates,values):row['rerank_score']=float(value)
                candidates.sort(key=lambda row:(-row['rerank_score'],row['citation_id']))
                if self.min_rerank_score is not None:
                    candidates=[row for row in candidates if row['rerank_score']>=self.min_rerank_score]
            reranked_at=time.perf_counter()
            selected=[];parts=[];seen=set()
            for row in candidates:
                digest=sha(re.sub(r'\s+',' ',row['body']).strip())
                if digest in seen:continue
                # Avoid spending context twice on overlapping pieces of a file.
                if any(other['doc_key']==row['doc_key'] and
                       max(0,min(other['end'],row['end'])-max(other['start'],row['start']))
                       >=.5*min(other['end']-other['start'],row['end']-row['start']) for other in selected):continue
                part=render(row)
                if self.count('\n\n'.join(parts+[part]))>context_budget:continue
                parts.append(part);selected.append(row);seen.add(digest)
                if len(selected)==top_k:break
            context='\n\n'.join(parts)
            citations=[{key:row[key] for key in ['citation_id','document_id','namespace','title','source_uri','source_revision',
                        'revision_hash','start','end','platforms','versions','rrf_score','lexical_score','vector_score','rerank_score'] if key in row}
                       for row in selected]
            finished=time.perf_counter()
            return {'status':'candidate_evidence' if selected else ('context_budget_too_small' if candidates else 'no_evidence'),
                'context':context,'citations':citations,'candidate_count':len(candidates),
                'budget':{'limit':context_budget,'used':int(self.count(context)),'tokenizer':self.tokenizer_name,'scope':'rendered_context'},
                'timing':{'lexical_s':lexical_at-started,'vector_s':semantic_at-lexical_at,
                          'fusion_s':ranked_at-semantic_at,'rerank_s':reranked_at-ranked_at,
                          'context_s':finished-reranked_at,'total_s':finished-started},
                'mode':mode,'reranked':rerank,'answerability':'not inferred from relevance scores'}

    def read_evidence(self,citation_id,context_budget=4096):
        positive(context_budget,'context_budget',32768)
        with self.lock,self.store.snapshot():
            row=self.store.read(citation_id,self.principal,self.namespaces)
            context=render(row)
            count=int(self.count(context))
            if count>context_budget:
                return {'status':'context_budget_too_small','citation_id':citation_id,
                        'required':count,'budget':context_budget,'tokenizer':self.tokenizer_name}
            return {'status':row['status'],'is_current':row['is_current'],'context':context,
                'citation':{key:row[key] for key in ['citation_id','document_id','namespace','source_uri','source_revision','start','end','revision_hash']},
                'budget':{'limit':context_budget,'used':count,'tokenizer':self.tokenizer_name,'scope':'rendered_context'}}
