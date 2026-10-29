"""Offline BGE cross-encoder scoring on the InfiniTensor device runtime."""
import json
from pathlib import Path
import time

import numpy as np

from . import backend
from .batching import plan_batches, positive_integer
from .embedding import fp32_inference_policy, _integer
from .onnx import OnnxStub
from .model_package import package_identity


def position_ids(input_ids, padding_idx=1):
    """Match XLM-R's position preprocessing, including right padding.

    This is token preprocessing, not encoder execution. The complete encoder
    and classification head remain in the device graph.
    """
    mask = (input_ids != padding_idx).astype(np.int64)
    return np.cumsum(mask, axis=1, dtype=np.int64) * mask + padding_idx


class RerankerSession:
    def __init__(self, model_dir, device='nvidia', device_id=0,
                 max_batch_size=8, max_length=512, batching='length', token_budget=4096):
        import onnx
        from transformers import AutoTokenizer

        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / 'model.onnx'
        self.max_batch_size = positive_integer(max_batch_size, 'max_batch_size')
        self.max_length = positive_integer(max_length, 'max_length')
        self.token_budget = positive_integer(token_budget, 'token_budget')
        if self.max_batch_size > 32 or not 4 <= self.max_length <= 512:
            raise ValueError('BGE reranker supports batch <= 32 and 4 <= max_length <= 512')
        if self.token_budget < self.max_length:
            raise ValueError('token_budget must accommodate max_length')
        if batching not in ('input', 'length'):
            raise ValueError('batching must be input or length')
        self.batching = batching
        manifest = json.loads((self.model_dir / 'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('task') != 'reranking' or manifest.get('model') != 'BAAI/bge-reranker-base':
            raise ValueError('expected a frozen BAAI/bge-reranker-base package')
        if manifest.get('export_validation', {}).get('status') != 'PASS':
            raise ValueError('reranker package has no passing export reference validation')
        if 'model.onnx' not in manifest.get('sha256', {}):
            raise ValueError('reranker package lacks its ONNX hash')
        if manifest.get('precision') != 'FP32':
            raise ValueError('reranker requires a frozen FP32 model package')
        self.package_identity = package_identity(self.model_dir, self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = 'right'
        self.stub = OnnxStub(onnx.load(str(self.model_path)),
            backend.runtime(device, _integer(device_id, "device_id")))
        expected = {'input_ids', 'attention_mask', 'position_ids'}
        if set(self.stub.inputs) != expected or set(self.stub.outputs) != {'scores'}:
            raise ValueError('reranker graph must accept token/mask/position IDs and return scores')
        self.last_timing = {}

    def score(self, pairs, batch_size=None, batching=None):
        pairs = list(pairs)
        size = self.max_batch_size if batch_size is None else positive_integer(batch_size, 'batch_size')
        if size > self.max_batch_size:
            raise ValueError('batch_size exceeds the session capacity')
        if any(not isinstance(pair, (list, tuple)) or len(pair) != 2
               or any(not isinstance(text, str) for text in pair) for pair in pairs):
            raise TypeError('pairs must contain (query, passage) strings')
        order = self.batching if batching is None else batching
        if order not in ('input', 'length'):
            raise ValueError('batching must be input or length')
        if not pairs:
            self.last_timing = {'total_s': 0., 'batches': 0, 'padded_tokens': 0, 'effective_tokens': 0, 'input_shapes': []}
            return np.empty((0,), dtype=np.float32)
        started = time.perf_counter()
        encoded = self.tokenizer([pair[0] for pair in pairs], [pair[1] for pair in pairs],
            padding=False, truncation=True, max_length=self.max_length,
            return_attention_mask=True, return_token_type_ids=False)
        tokenized = time.perf_counter()
        lengths = [len(ids) for ids in encoded['input_ids']]
        batches = plan_batches(lengths, size, self.token_budget, order)
        planned = time.perf_counter()
        scores = np.empty(len(pairs), dtype=np.float32)
        timing = {'tokenize_s': tokenized-started, 'plan_s': planned-tokenized,
                  'pack_s': 0., 'prepare_s': 0., 'h2d_s': 0., 'run_s': 0., 'd2h_s': 0.,
                  'restore_s': 0., 'padded_tokens': 0, 'effective_tokens': sum(lengths),
                  'batches': len(batches), 'input_shapes': [], 'batching': order}
        with fp32_inference_policy():
            for indices in batches:
                packed_at = time.perf_counter()
                shape = (len(indices), max(lengths[index] for index in indices))
                ids = np.full(shape, self.tokenizer.pad_token_id, dtype=np.int64)
                mask = np.zeros(shape, dtype=np.int64)
                for row, index in enumerate(indices):
                    length = lengths[index]
                    ids[row, :length] = encoded['input_ids'][index]
                    mask[row, :length] = encoded['attention_mask'][index]
                positions = position_ids(ids, self.tokenizer.pad_token_id)
                timing['pack_s'] += time.perf_counter() - packed_at
                values = self.stub.run_numpy({'input_ids': ids, 'attention_mask': mask,
                                              'position_ids': positions})['scores']
                restored_at = time.perf_counter()
                values = np.asarray(values)
                if values.dtype != np.float32 or values.shape not in ((len(indices),), (len(indices), 1)) or not np.isfinite(values).all():
                    raise RuntimeError('reranker returned invalid scores')
                scores[np.asarray(indices)] = values.reshape(-1)
                timing['restore_s'] += time.perf_counter() - restored_at
                timing['input_shapes'].append(list(shape))
                timing['padded_tokens'] += shape[0] * shape[1]
                for key in ('prepare_s', 'h2d_s', 'run_s', 'd2h_s'):
                    timing[key] += self.stub.last_timing[key]
        timing['total_s'] = time.perf_counter() - started
        self.last_timing = timing
        return scores

    def rerank(self, query, documents, top_k=10, ids=None, batch_size=None):
        if not isinstance(query, str):
            raise TypeError('query must be a string')
        documents = list(documents)
        if any(not isinstance(text, str) for text in documents):
            raise TypeError('documents must be strings')
        top_k = positive_integer(top_k, 'top_k')
        if ids is None:
            tie = np.arange(len(documents))
        else:
            ids = list(ids)
            if len(ids) != len(documents) or any(not isinstance(value, str) for value in ids) or len(set(ids)) != len(ids):
                raise ValueError('document ids must be unique strings matching documents')
            tie = np.asarray(ids, dtype=str)
        scores = self.score([(query, text) for text in documents], batch_size=batch_size)
        order = np.lexsort((tie, -scores))[:min(top_k, len(documents))]
        return [{'index': int(index), 'id': ids[index] if ids is not None else str(index),
                 'score': float(scores[index])} for index in order]
