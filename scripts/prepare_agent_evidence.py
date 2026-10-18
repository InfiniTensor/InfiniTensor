#!/usr/bin/env python3
"""Freeze real repository documents and public issue-title queries offline."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def digest(data):return hashlib.sha256(data).hexdigest()


# Relevant-document annotations, not claims that the historical bug is fixed.
# Query strings are taken verbatim from the supplied public issue snapshot.
RELEVANCE={
    242:['docs/INSTALL_GUIDE_CN.md'],
    200:['README.md','README_CN.md','docs/INSTALL_GUIDE_CN.md'],
    296:['docs/BGE_EMBEDDING_M1_CN.md','docs/KNOWLEDGE_RETRIEVAL_VALIDATION_CN.md'],
    145:['README.md','docs/USER_GUIDE_CN.md'],
    144:['docs/INDEX.md'],
    161:['docs/INSTALL_GUIDE_CN.md'],
    247:['docs/BGE_EMBEDDING_M1_CN.md'],
    112:['docs/BGE_EMBEDDING_M1_CN.md','docs/KNOWLEDGE_RETRIEVAL_CN.md'],
    70:['docs/USER_GUIDE_CN.md','docs/BGE_EMBEDDING_M1_CN.md'],
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--revision',required=True)
    parser.add_argument('--issues',required=True,help='fixed GitHub issues API snapshot with issues array')
    parser.add_argument('--output',required=True);args=parser.parse_args()
    root=Path(args.source);output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    revision=subprocess.check_output(['git','rev-parse',args.revision],cwd=root,text=True).strip()
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',revision],cwd=root,text=True).splitlines()
    paths=[p for p in paths if (p.startswith('docs/') and p.endswith('.md')) or p in ['README.md','README_CN.md','dependencies/m1-lock.json','dependencies/retrieval-lock.json']]
    documents=[]
    for path in sorted(paths):
        raw=subprocess.check_output(['git','show',revision+':'+path],cwd=root)
        text=raw.decode('utf-8-sig')
        # Heading sections retain the exact source slice and a permalink to
        # its starting line. Each section is a versioned logical document.
        boundaries=[0]+[m.start() for m in re.finditer(r'^#{1,3} .+',text,re.M) if m.start()>0]+[len(text)]
        for start,end in zip(boundaries,boundaries[1:]):
            body=text[start:end]
            if not body.strip():continue
            heading=body.splitlines()[0].lstrip('# ').strip()[:60] or path
            line=text[:start].count('\n')+1
            documents.append({'id':path+':L'+str(line),'namespace':'infinitensor','title':heading,
                'text':body,'source_uri':f'https://github.com/InfiniTensor/InfiniTensor/blob/{revision}/{path}#L{line}',
                'revision':revision,'platforms':['any'],'versions':{'InfiniTensor':[revision]},'readers':['*'],
                'metadata':{'source_path':path,'source_sha256':digest(raw),'source_start_char':start,'source_start_line':line}})
    issues_raw=Path(args.issues).read_bytes();snapshot=json.loads(issues_raw)
    queries=[]
    for issue in sorted(snapshot['issues'],key=lambda row:row['number']):
        title=issue['title'].strip()
        if not title:continue
        queries.append({'id':'issue-'+str(issue['number']),'query':title,'source_uri':issue['html_url'],
            'created_at':issue['created_at'],'updated_at':issue['updated_at'],
            'relevant_paths':RELEVANCE.get(issue['number'],[]),
            'label_scope':'curated related-document guidance, not a verified solution' if issue['number'] in RELEVANCE else 'unlabeled performance query'})
    for name,rows in [('documents.jsonl',documents),('queries.jsonl',queries)]:
        (output/name).write_text(''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in rows),encoding='utf-8')
    manifest={'source_revision':revision,'source_files':len(paths),'documents':len(documents),'real_queries':len(queries),
        'labeled_queries':sum(bool(q['relevant_paths']) for q in queries),'issues_sha256':digest(issues_raw),
        'sha256':{name:digest((output/name).read_bytes()) for name in ['documents.jsonl','queries.jsonl']},
        'selection':'all nonempty issue titles from fixed snapshot; corpus is repository documentation only',
        'leakage_control':'issue questions, bodies and comments are never included in the search corpus',
        'scope':'small real-query retrieval pilot; no generated questions, no LLM task-success claim'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
