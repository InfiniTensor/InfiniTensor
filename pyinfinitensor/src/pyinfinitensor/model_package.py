"""Identity of the executable model and its offline preprocessing files."""
import hashlib
import json
from pathlib import Path


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def package_identity(model_dir, model_path):
    folder = Path(model_dir)
    manifest_path = folder/'manifest.json'
    expected = json.loads(manifest_path.read_text(encoding='utf-8')).get('sha256', {}) if manifest_path.exists() else {}
    names = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
             'vocab.txt', 'vocab.json', 'merges.txt', 'sentencepiece.bpe.model', 'added_tokens.json']
    files = {Path(model_path).name: file_digest(model_path)}
    for name in names:
        path = folder/name
        if path.is_file():
            files[name] = file_digest(path)
        elif name in expected:
            raise ValueError('model package is missing '+name)
    for name, digest in files.items():
        if name in expected and expected[name] != digest:
            raise ValueError('model package hash mismatch: '+name)
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {'fingerprint': fingerprint, 'sha256': files}
