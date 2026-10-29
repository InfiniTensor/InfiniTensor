"""Offline administration, search and read-only MCP stdio entry points."""
import argparse
import json
from pathlib import Path

from .store import EvidenceStore
from .service import EvidenceService


def create_engine(args):
    if args.lexical_only:return None
    if not args.embedding_model or not args.reranker_model:
        raise ValueError('InfiniTensor mode requires both frozen local model packages')
    from .engine import InfiniTensorEngine
    return InfiniTensorEngine(args.embedding_model,args.reranker_model,args.device,args.device_id,
        args.batch_size,cache=not args.no_cache)


def create_service(args,read_only=True):
    engine=create_engine(args)
    counter=None;tokenizer_name=None
    if args.context_tokenizer:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(args.context_tokenizer,local_files_only=True)
        counter=lambda text:len(tokenizer(text,add_special_tokens=False,truncation=False,verbose=False)['input_ids'])
        tokenizer_name=str(Path(args.context_tokenizer).resolve())
    return EvidenceService(EvidenceStore(args.store,read_only=read_only),args.principal,args.namespace,
        engine,counter,tokenizer_name,args.min_rerank_score)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['build','withdraw','search','read','serve'])
    p.add_argument('--store',required=True);p.add_argument('--documents')
    p.add_argument('--sync-namespace',help='explicitly replace this namespace snapshot, withdrawing absent documents')
    p.add_argument('--namespace',action='append',required=True,help='host-authorized namespace; repeat to allow more')
    p.add_argument('--principal',required=True,help='host identity, never taken from tool arguments')
    p.add_argument('--id');p.add_argument('--query');p.add_argument('--citation')
    p.add_argument('--platform',choices=['nvidia','cambricon','ascend','cpu'])
    p.add_argument('--version',action='append',default=[],help='exact Component=Version; repeat for dependencies')
    p.add_argument('--top-k',type=int,default=5);p.add_argument('--recall-k',type=int,default=50)
    p.add_argument('--context-budget',type=int,default=2048)
    p.add_argument('--lexical-only',action='store_true',help='explicit CPU BM25 baseline, with no inference fallback')
    p.add_argument('--mode',choices=['lexical','vector','hybrid'],default='hybrid')
    p.add_argument('--rerank',action='store_true',help='opt in after evaluating domain relevance; generic reranking can regress')
    p.add_argument('--no-cache',action='store_true')
    p.add_argument('--embedding-model');p.add_argument('--reranker-model');p.add_argument('--context-tokenizer')
    p.add_argument('--device',choices=['nvidia','cambricon','ascend'],default='nvidia')
    p.add_argument('--device-id',type=int,default=0);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--min-rerank-score',type=float,help='optional application-calibrated threshold; logits are not probabilities')
    return p


def main():
    p=parser();args=p.parse_args()
    if args.command=='serve':
        from .mcp_server import serve
        serve(args);return
    if args.command=='withdraw':
        if not args.id or len(args.namespace)!=1:p.error('withdraw requires --id and exactly one namespace')
        store=EvidenceStore(args.store)
        try:print(json.dumps({'withdrawn':store.withdraw(args.namespace[0],args.id)}))
        finally:store.close()
        return
    if args.command=='build':
        if not args.documents:p.error('build requires --documents')
        values=[json.loads(line) for line in Path(args.documents).read_text(encoding='utf-8').splitlines() if line.strip()]
        if any(value.get('namespace') not in args.namespace for value in values):p.error('input document is outside the supplied namespaces')
        engine=create_engine(args);store=EvidenceStore(args.store)
        try:
            kwargs={'encode':engine.encode_documents,'fingerprint':engine.fingerprint,'chunker':engine.chunks} if engine else {}
            result=store.upsert(values,**kwargs,sync_namespace=args.sync_namespace)
            result.update(status='PASS',backend='InfiniTensor' if engine else 'CPU lexical baseline')
            print(json.dumps(result,ensure_ascii=False))
        finally:store.close()
        return
    service=create_service(args)
    try:
        versions={}
        for value in args.version:
            if '=' not in value:p.error('version must be Component=Version')
            component,version=value.split('=',1);versions[component]=version
        filters={'versions':versions}
        if args.platform:filters['platform']=args.platform
        if args.command=='read':
            if not args.citation:p.error('read requires --citation')
            result=service.read_evidence(args.citation,args.context_budget,filters)
        else:
            if args.query is None:p.error('search requires --query')
            result=service.search_evidence(args.query,filters,args.top_k,args.recall_k,args.context_budget,
                mode='lexical' if args.lexical_only else args.mode,rerank=args.rerank and not args.lexical_only)
        print(json.dumps(result,ensure_ascii=False))
    finally:service.store.close()


if __name__=='__main__':main()
