"""Transactional SQLite evidence snapshots, filtered before candidate selection."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time

import numpy as np

from .text import character_chunks,positive,terms


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))


def sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def strings(value,name):
    if not isinstance(value,(list,tuple)) or not value or any(not isinstance(x,str) or not x for x in value):
        raise ValueError(name+' must be nonempty strings')
    return sorted(set(value))


def document(value):
    if not isinstance(value,dict):raise TypeError('document must be an object')
    allowed={'id','namespace','title','text','source_uri','revision','platforms','versions','readers','metadata'}
    if set(value)-allowed:raise ValueError('unknown document fields: '+str(sorted(set(value)-allowed)))
    result=dict(value)
    for key in ['id','namespace','text','source_uri','revision']:
        if not isinstance(result.get(key),str) or not result[key].strip():raise ValueError(key+' must be a nonempty string')
    result.setdefault('title',result['id'])
    if not isinstance(result['title'],str):raise ValueError('title must be a string')
    result['platforms']=strings(result.get('platforms',['any']),'platforms')
    if set(result['platforms'])-{'any','nvidia','cambricon','ascend','cpu'}:raise ValueError('unknown platform')
    result['readers']=strings(result.get('readers',['*']),'readers')
    versions=result.get('versions',{})
    if not isinstance(versions,dict) or any(not isinstance(k,str) or not k for k in versions):raise ValueError('invalid version map')
    result['versions']={key:strings(val,'version values') for key,val in versions.items()}
    result.setdefault('metadata',{})
    if not isinstance(result['metadata'],dict):raise ValueError('metadata must be an object')
    canonical(result)
    return result


class EvidenceStore:
    def __init__(self,path,read_only=False):
        self.path=Path(path).resolve();self.lock=threading.RLock();self.read_only=read_only
        if read_only:
            self.db=sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True,isolation_level=None,check_same_thread=False)
        else:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            self.db=sqlite3.connect(str(self.path),isolation_level=None,check_same_thread=False)
        self.db.row_factory=sqlite3.Row
        self.db.execute('PRAGMA busy_timeout=10000')
        self.db.execute('PRAGMA foreign_keys=ON')
        if not read_only:
            self.db.execute('PRAGMA journal_mode=WAL')
            self.db.executescript('''
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents(doc_key TEXT PRIMARY KEY,namespace TEXT NOT NULL,doc_id TEXT NOT NULL,
 current_revision TEXT NOT NULL,withdrawn INTEGER NOT NULL DEFAULT 0,readers TEXT NOT NULL,
 platforms TEXT NOT NULL,versions TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS revisions(doc_key TEXT NOT NULL,revision_hash TEXT NOT NULL,payload TEXT NOT NULL,
 created REAL NOT NULL,PRIMARY KEY(doc_key,revision_hash));
CREATE TABLE IF NOT EXISTS chunks(rowid INTEGER PRIMARY KEY,citation_id TEXT UNIQUE NOT NULL,
 doc_key TEXT NOT NULL,revision_hash TEXT NOT NULL,start INTEGER NOT NULL,end INTEGER NOT NULL,
 body TEXT NOT NULL,embedding_text TEXT NOT NULL,vector BLOB,dimension INTEGER);
CREATE INDEX IF NOT EXISTS chunks_document ON chunks(doc_key,revision_hash);
CREATE VIRTUAL TABLE IF NOT EXISTS lexical USING fts5(title_terms,body_terms,tokenize='ascii');
''')
            self.db.execute("INSERT OR IGNORE INTO settings VALUES('schema','1')")
        if self.setting('schema')!='1':raise ValueError('unsupported evidence store schema')

    def setting(self,key):
        with self.lock:
            row=self.db.execute('SELECT value FROM settings WHERE key=?',(key,)).fetchone()
            return row[0] if row else None

    def close(self):
        with self.lock:self.db.close()

    @contextmanager
    def snapshot(self):
        """One read snapshot for filtering, ranking and quote assembly."""
        with self.lock:
            if self.db.in_transaction:
                yield
                return
            self.db.execute('BEGIN')
            try:yield
            finally:self.db.execute('ROLLBACK')

    @contextmanager
    def transaction(self):
        if self.read_only:raise PermissionError('evidence store is read-only')
        with self.lock:
            self.db.execute('BEGIN IMMEDIATE')
            try:
                yield
                self.db.execute('COMMIT')
            except BaseException:
                self.db.execute('ROLLBACK');raise

    def upsert(self,documents,encode=None,fingerprint=None,chunker=character_chunks,sync_namespace=None):
        if self.read_only:raise PermissionError('evidence store is read-only')
        values=[document(x) for x in documents]
        keys=[sha(canonical([x['namespace'],x['id']])) for x in values]
        if sync_namespace is not None and (not isinstance(sync_namespace,str) or not sync_namespace or
                any(x['namespace']!=sync_namespace for x in values)):
            raise ValueError('sync requires one explicitly matching namespace')
        if len(set(keys))!=len(keys):raise ValueError('duplicate document in update')
        if (encode is None)!=(fingerprint is None):raise ValueError('encode and fingerprint must be supplied together')
        # Keep model work outside the write transaction. A failed encode must
        # not publish a partially indexed revision or remove existing results.
        with self.lock:
            bound=self.setting('embedding_fingerprint')
            if bound is not None and fingerprint!=bound:raise ValueError('embedding package differs; build a new store')
            def namespace_state():
                return [tuple(row) for row in self.db.execute(
                    'SELECT doc_key,current_revision,withdrawn FROM documents WHERE namespace=? ORDER BY doc_key',
                    (sync_namespace,))]
            observed_namespace=namespace_state() if sync_namespace is not None else None
            prepared=[];observed={}
            for value,key in zip(values,keys):
                revision=sha(canonical(value));spans=chunker(value['text'])
                chunks=[]
                for start,end in spans:
                    if not isinstance(start,int) or not isinstance(end,int) or not 0<=start<end<=len(value['text']):raise ValueError('invalid source span')
                    body=value['text'][start:end]
                    chunks.append({'start':start,'end':end,'body':body,
                        'embedding_text':value['title']+'\n'+body,
                        'citation_id':sha(canonical([key,revision,start,end]))})
                current=self.db.execute('SELECT current_revision,withdrawn FROM documents WHERE doc_key=?',(key,)).fetchone()
                observed[key]=tuple(current) if current else None
                if current and current[0]==revision and not current[1]:continue
                prepared.append((value,key,revision,chunks))
            texts=[chunk['embedding_text'] for _,_,_,chunks in prepared for chunk in chunks]
            vectors=None
            if encode and texts:
                vectors=np.ascontiguousarray(encode(texts),dtype=np.float32)
                if vectors.ndim!=2 or vectors.shape[0]!=len(texts) or vectors.shape[1]<1 or not np.isfinite(vectors).all():raise ValueError('invalid embedding output')
                if not np.allclose(np.linalg.norm(vectors,axis=1),1.,atol=1e-5,rtol=0):raise ValueError('embeddings must be L2-normalized')
                old_dimension=self.setting('dimension')
                if old_dimension and int(old_dimension)!=vectors.shape[1]:raise ValueError('embedding dimension changed')
            with self.transaction():
                # Another process may have opened the same database while this
                # encoder was working. Recheck its package inside the commit.
                bound=self.setting('embedding_fingerprint')
                if bound is not None and fingerprint!=bound:raise ValueError('embedding package changed concurrently')
                if observed_namespace is not None and namespace_state()!=observed_namespace:
                    raise RuntimeError('namespace changed concurrently; retry the complete update')
                for key,expected in observed.items():
                    current=self.db.execute('SELECT current_revision,withdrawn FROM documents WHERE doc_key=?',(key,)).fetchone()
                    if (tuple(current) if current else None)!=expected:
                        raise RuntimeError('document changed concurrently; retry the complete update')
                if vectors is not None:
                    dimension=self.setting('dimension')
                    if dimension is not None and int(dimension)!=vectors.shape[1]:
                        raise ValueError('embedding dimension changed concurrently')
                    self.db.execute("INSERT OR REPLACE INTO settings VALUES('embedding_fingerprint',?)",(fingerprint,))
                    self.db.execute("INSERT OR REPLACE INTO settings VALUES('dimension',?)",(str(vectors.shape[1]),))
                offset=0;withdrawn=0
                for value,key,revision,chunks in prepared:
                    self.db.execute('DELETE FROM lexical WHERE rowid IN (SELECT rowid FROM chunks WHERE doc_key=?)',(key,))
                    self.db.execute('INSERT OR IGNORE INTO revisions VALUES(?,?,?,?)',(key,revision,canonical(value),time.time()))
                    self.db.execute('INSERT OR REPLACE INTO documents VALUES(?,?,?,?,0,?,?,?)',
                        (key,value['namespace'],value['id'],revision,canonical(value['readers']),canonical(value['platforms']),canonical(value['versions'])))
                    for chunk in chunks:
                        vector=vectors[offset].tobytes() if vectors is not None else None
                        dimension=vectors.shape[1] if vectors is not None else None
                        self.db.execute('INSERT OR REPLACE INTO chunks(citation_id,doc_key,revision_hash,start,end,body,embedding_text,vector,dimension) VALUES(?,?,?,?,?,?,?,?,?)',
                            (chunk['citation_id'],key,revision,chunk['start'],chunk['end'],chunk['body'],chunk['embedding_text'],vector,dimension))
                        rowid=self.db.execute('SELECT rowid FROM chunks WHERE citation_id=?',(chunk['citation_id'],)).fetchone()[0]
                        self.db.execute('INSERT INTO lexical(rowid,title_terms,body_terms) VALUES(?,?,?)',
                            (rowid,' '.join(terms(value['title'])),' '.join(terms(chunk['body']))))
                        offset+=1
                if sync_namespace is not None:
                    present=set(keys)
                    for row in self.db.execute('SELECT doc_key FROM documents WHERE namespace=? AND withdrawn=0',(sync_namespace,)).fetchall():
                        if row[0] not in present:
                            self.db.execute('DELETE FROM lexical WHERE rowid IN (SELECT rowid FROM chunks WHERE doc_key=?)',(row[0],))
                            self.db.execute('UPDATE documents SET withdrawn=1 WHERE doc_key=?',(row[0],));withdrawn+=1
            return {'updated_documents':len(prepared),'encoded_chunks':len(texts) if vectors is not None else 0,
                    'chunks':sum(len(x[3]) for x in prepared),'withdrawn_documents':withdrawn}

    def withdraw(self,namespace,doc_id):
        key=sha(canonical([namespace,doc_id]))
        with self.transaction():
            self.db.execute('DELETE FROM lexical WHERE rowid IN (SELECT rowid FROM chunks WHERE doc_key=?)',(key,))
            changed=self.db.execute('UPDATE documents SET withdrawn=1 WHERE doc_key=?',(key,)).rowcount
        return bool(changed)

    def _scope(self,principal,namespaces,filters=None):
        if not isinstance(principal,str) or not principal:raise ValueError('principal is required')
        namespaces=strings(namespaces,'allowed namespaces')
        filters={} if filters is None else filters
        if not isinstance(filters,dict) or set(filters)-{'platform','versions'}:raise ValueError('unknown filters')
        sql=['d.withdrawn=0','d.namespace IN ('+','.join('?' for _ in namespaces)+')',
             "EXISTS (SELECT 1 FROM json_each(d.readers) WHERE value='*' OR value=?)"]
        params=list(namespaces)+[principal]
        if filters.get('platform') is not None:
            platform=filters['platform']
            if platform not in {'nvidia','cambricon','ascend','cpu'}:raise ValueError('unknown platform filter')
            sql.append("EXISTS (SELECT 1 FROM json_each(d.platforms) WHERE value='any' OR value=?)");params.append(platform)
        versions=filters.get('versions',{})
        if not isinstance(versions,dict):raise ValueError('versions must be an exact-version object')
        for component,version in sorted(versions.items()):
            if not isinstance(component,str) or not component or not isinstance(version,str) or not version:raise ValueError('invalid version filter')
            sql.append("(NOT EXISTS (SELECT 1 FROM json_each(d.versions) WHERE key=?) OR EXISTS (SELECT 1 FROM json_each(d.versions) v,json_each(v.value) a WHERE v.key=? AND (a.value='*' OR a.value=?)))")
            params.extend([component,component,version])
        return ' AND '.join(sql),params

    def _row(self,row):
        result=dict(row);payload=json.loads(result.pop('payload'))
        result.update(title=payload['title'],source_uri=payload['source_uri'],source_revision=payload['revision'],
            namespace=payload['namespace'],document_id=payload['id'],platforms=payload['platforms'],versions=payload['versions'])
        return result

    def lexical_search(self,query,principal,namespaces,filters=None,limit=50):
        positive(limit,'limit',200);tokens=list(dict.fromkeys(terms(query)))[:128]
        if not tokens:return []
        scope,params=self._scope(principal,namespaces,filters)
        sql='''SELECT c.*,r.payload,bm25(lexical,3.0,1.0) AS lexical_score FROM lexical
 JOIN chunks c ON c.rowid=lexical.rowid JOIN documents d ON d.doc_key=c.doc_key
 JOIN revisions r ON r.doc_key=c.doc_key AND r.revision_hash=c.revision_hash
 WHERE lexical MATCH ? AND d.current_revision=c.revision_hash AND '''+scope+' ORDER BY lexical_score,c.citation_id LIMIT ?'
        with self.lock:
            return [self._row(x) for x in self.db.execute(sql,[' OR '.join(tokens),*params,limit])]

    def eligible_chunks(self,principal,namespaces,filters=None):
        scope,params=self._scope(principal,namespaces,filters)
        sql='''SELECT c.*,r.payload FROM chunks c JOIN documents d ON d.doc_key=c.doc_key
 JOIN revisions r ON r.doc_key=c.doc_key AND r.revision_hash=c.revision_hash
 WHERE d.current_revision=c.revision_hash AND '''+scope+' ORDER BY c.citation_id'
        with self.lock:return [self._row(row) for row in self.db.execute(sql,params)]

    def read(self,citation_id,principal,namespaces,filters=None):
        if not isinstance(citation_id,str) or len(citation_id)!=64:raise ValueError('invalid citation id')
        scope,params=self._scope(principal,namespaces,filters)
        sql='''SELECT c.*,r.payload,d.current_revision FROM chunks c JOIN documents d ON d.doc_key=c.doc_key
 JOIN revisions r ON r.doc_key=c.doc_key AND r.revision_hash=c.revision_hash
 WHERE c.citation_id=? AND '''+scope
        with self.lock:row=self.db.execute(sql,[citation_id,*params]).fetchone()
        if row is None:raise KeyError('evidence is unavailable in this scope')
        result=self._row(row)
        # A historical quote must also match the host environment itself, even
        # when the latest document has since moved to another platform/version.
        filters=filters or {}
        if filters.get('platform') and not {'any',filters['platform']}.intersection(result['platforms']):
            raise KeyError('evidence is unavailable in this scope')
        for component,version in filters.get('versions',{}).items():
            allowed=result['versions'].get(component)
            if allowed is not None and '*' not in allowed and version not in allowed:
                raise KeyError('evidence is unavailable in this scope')
        result['is_current']=result.pop('current_revision')==result['revision_hash']
        result['status']='current' if result['is_current'] else 'superseded'
        result.pop('vector',None);result.pop('embedding_text',None)
        return result
