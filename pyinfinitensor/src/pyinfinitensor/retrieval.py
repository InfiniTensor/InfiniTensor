"""A persistent local document index with InfiniTensor recall and reranking."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .embedding import cosine_top_k


def model_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


class KnowledgeIndex:
    def __init__(self, ids, texts, vectors, metadata):
        self.ids, self.texts = list(ids), list(texts)
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.metadata = dict(metadata)
        if (len(self.ids) != len(self.texts) or len(set(self.ids)) != len(self.ids)
                or any(not isinstance(value, str) for value in self.ids + self.texts)
                or self.vectors.shape != (len(self.ids), 512) or not np.isfinite(self.vectors).all()):
            raise ValueError('invalid document index')
        if len(self.ids) and not np.allclose(np.linalg.norm(self.vectors, axis=1), 1., atol=1e-5, rtol=0):
            raise ValueError('index vectors must be L2-normalized')

    @classmethod
    def build(cls, rows, embedding_session, batch_size=32):
        rows = list(rows)
        if any(not isinstance(row, dict) or not isinstance(row.get('id'), str)
               or not isinstance(row.get('text'), str) for row in rows):
            raise ValueError('documents require string id and text')
        ids, texts = [r['id'] for r in rows], [r['text'] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate document id')
        vectors = embedding_session.encode(texts, kind='passage', batch_size=batch_size)
        metadata = {'schema_version': 1, 'model_onnx_sha256': embedding_session.package_identity['sha256'][embedding_session.model_path.name],
            'embedding_package_fingerprint': embedding_session.package_identity['fingerprint'],
            'preprocessing': 'BGE Chinese query instruction; CLS; L2; right padding; truncate 512',
            'documents_sha256': hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
            'rows': len(ids), 'dimension': 512, 'kind': 'passage'}
        return cls(ids, texts, vectors, metadata)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # A single self-contained artifact keeps vectors, text and IDs aligned.
        temporary = path.with_name(path.name + '.tmp')
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, ids=np.asarray(self.ids, dtype=str),
                texts=np.asarray(self.texts, dtype=str), vectors=self.vectors,
                metadata=json.dumps(self.metadata, ensure_ascii=False))
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as values:
            return cls(values['ids'].tolist(), values['texts'].tolist(), values['vectors'],
                       json.loads(str(values['metadata'])))

    def search(self, query, embedding_session, reranker_session, recall_k=50, top_k=10):
        if not isinstance(query, str):
            raise TypeError('query must be a string')
        from .batching import positive_integer
        recall_k = positive_integer(recall_k, 'recall_k')
        top_k = positive_integer(top_k, 'top_k')
        if top_k > recall_k:
            raise ValueError('top_k cannot exceed recall_k')
        if not self.ids:
            return {'results': [], 'timing': {'total_s': 0.}}
        if embedding_session.package_identity['fingerprint'] != self.metadata.get('embedding_package_fingerprint'):
            raise ValueError('embedding model/tokenizer package does not match the document index')
        # Cache the package identity on the long-lived session, not per query.
        identity = getattr(embedding_session, '_retrieval_model_digest', None)
        if identity is None:
            identity = model_digest(embedding_session.model_path)
            embedding_session._retrieval_model_digest = identity
        if identity != self.metadata['model_onnx_sha256']:
            raise ValueError('query embedding model does not match the document index')
        started = time.perf_counter()
        vector = embedding_session.encode([query], kind='query', batch_size=1)[0]
        encoded = time.perf_counter()
        recalled = cosine_top_k(vector, self.vectors, min(recall_k, len(self.ids)), self.ids)
        retrieved = time.perf_counter()
        indices = [item[0] for item in recalled]
        ranked = reranker_session.rerank(query, [self.texts[i] for i in indices],
            top_k=top_k, ids=[self.ids[i] for i in indices])
        finished = time.perf_counter()
        results = []
        for rank in ranked:
            candidate = rank['index']
            index = indices[candidate]
            results.append({'id': self.ids[index], 'text': self.texts[index],
                'recall_score': recalled[candidate][1], 'rerank_score': rank['score']})
        return {'results': results, 'timing': {'embedding_s': encoded-started,
            'retrieval_s': retrieved-encoded, 'reranking_s': finished-retrieved,
            'total_s': finished-started}, 'reranker': dict(reranker_session.last_timing)}
