#!/usr/bin/env python3
"""Export a verified local BGE reranker as one dynamic FP32 ONNX graph."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='verified local model/tokenizer directory')
    parser.add_argument('--revision', required=True, help='immutable upstream commit')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    import numpy as np
    import onnx
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    source, output = Path(args.source).resolve(strict=True), Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError('choose a new model package directory')
    output.mkdir(parents=True)
    torch.manual_seed(315)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        source, local_files_only=True, attn_implementation='eager').eval().float()
    model.config.use_cache = False
    for path in source.iterdir():
        if path.is_file() and path.suffix in ('.json', '.safetensors', '.model', '.md'):
            try:
                os.link(path, output / path.name)
            except OSError:
                shutil.copyfile(path, output / path.name)

    class CrossEncoder(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids, attention_mask, position_ids):
            return self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, return_dict=False)[0].reshape(-1)

    wrapper = CrossEncoder(model).eval()
    pairs = [('什么是语义检索？', '向量检索通过语义表示查找相关文档。'), ('panda', '熊猫生活在中国。')]

    def inputs(items):
        values = tokenizer([p[0] for p in items], [p[1] for p in items],
            padding=True, truncation=True, max_length=512, return_tensors='pt',
            return_token_type_ids=False)
        ids = values['input_ids'].to(torch.int64)
        mask = values['attention_mask'].to(torch.int64)
        valid = (ids != tokenizer.pad_token_id).to(torch.int64)
        positions = torch.cumsum(valid, dim=1) * valid + tokenizer.pad_token_id
        return ids, mask, positions

    sample = inputs(pairs)
    path = output / 'model.onnx'
    torch.onnx.export(wrapper, sample, str(path),
        input_names=['input_ids', 'attention_mask', 'position_ids'], output_names=['scores'],
        dynamic_axes={name: {0: 'batch', 1: 'sequence'} for name in ('input_ids', 'attention_mask', 'position_ids')}
            | {'scores': {0: 'batch'}},
        opset_version=17, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(str(path))
    validation = []
    if args.validate:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        reference = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        cases = [pairs, [('', '')], [('动态输入', '长文档。' * 700)], pairs * 4]
        for case in cases:
            ids, mask, positions = inputs(case)
            with torch.inference_mode():
                canonical = model(input_ids=ids, attention_mask=mask).logits.reshape(-1).numpy()
                wrapped = wrapper(ids, mask, positions).numpy()
            actual = reference.run(None, dict(zip(('input_ids', 'attention_mask', 'position_ids'),
                                                 [x.numpy() for x in (ids, mask, positions)])))[0]
            np.testing.assert_allclose(wrapped, canonical, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(actual, canonical, atol=1e-4, rtol=1e-3)
            validation.append({'shape': list(ids.shape), 'max_abs_error': float(np.abs(actual-canonical).max())})
    graph = onnx.load(str(path))
    manifest = {'schema_version': 1, 'task': 'reranking', 'model': 'BAAI/bge-reranker-base',
        'revision': args.revision, 'opset': 17, 'precision': 'FP32', 'max_length': 512,
        'position_ids': 'CPU token preprocessing: cumsum(input_ids != pad) * mask + pad',
        'query_prefix': None, 'output': 'raw relevance logits; higher is more relevant',
        'versions': {'torch': torch.__version__, 'transformers': transformers.__version__, 'onnx': onnx.__version__},
        'export_validation': {'status': 'PASS' if args.validate else 'NOT_RUN', 'cases': validation},
        'nodes': [{'name': n.name, 'op': n.op_type, 'inputs': list(n.input), 'outputs': list(n.output)} for n in graph.graph.node],
        'sha256': {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'model_onnx_sha256': manifest['sha256']['model.onnx'],
        'nodes': len(graph.graph.node), 'operators': sorted({n.op_type for n in graph.graph.node}),
        'validation': validation}), flush=True)


if __name__ == '__main__':
    main()
