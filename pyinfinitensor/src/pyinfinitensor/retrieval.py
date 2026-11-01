"""Offline JSONL document encoding and deterministic local embedding search."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
from importlib import metadata

import numpy as np
from .embedding import cosine_top_k, BGE_QUERY_PREFIX, _integer


def _digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _documents(path):
    documents, ids = [], set()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(
                    "document must be a JSON object at line {}".format(number)
                )
            identifier = item.get("id")
            if isinstance(identifier, bool) or not isinstance(identifier, (str, int)):
                raise ValueError(
                    "document id must be a string or integer at line {}".format(number)
                )
            identifier = str(identifier)
            if (
                not identifier
                or identifier in ids
                or not isinstance(item.get("text"), str)
            ):
                raise ValueError(
                    "invalid or duplicate document at line {}".format(number)
                )
            ids.add(identifier)
            documents.append({"id": identifier, "text": item["text"]})
    if not documents:
        raise ValueError("document collection is empty")
    return documents


def _model_hash(session):
    name = str(session.model_path.relative_to(session.model_dir))
    return session.model_manifest["sha256"][name]


def _fingerprint(session):
    # A matching ONNX graph alone does not establish matching tokenization.
    tokenizer = getattr(session, "tokenizer", None)
    backend = getattr(tokenizer, "backend_tokenizer", None)
    tokenizer_state = json.loads(backend.to_str()) if backend is not None else None
    if tokenizer_state is not None:
        # Batch encoding changes these two fields; the fixed policy below owns
        # them, while vocab/normalizer/postprocessor changes remain significant.
        tokenizer_state.pop("padding", None)
        tokenizer_state.pop("truncation", None)
    versions = {}
    for name in ("transformers", "tokenizers"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    semantics = {
        "package": session.model_manifest["sha256"],
        "selected_onnx": str(session.model_path.relative_to(session.model_dir)),
        "selected_onnx_sha256": _model_hash(session),
        "tokenizer_state": tokenizer_state,
        "tokenizer_versions": versions,
        "padding": getattr(tokenizer, "padding_side", "right"),
        "truncation": getattr(tokenizer, "truncation_side", "right"),
        "model": session.model_manifest.get("model"),
        "revision": session.model_manifest["revision"],
        "max_length": session.max_length,
        "pooling": "CLS",
        "normalization": "L2",
        "dtype": "float32",
        "query_prefix": BGE_QUERY_PREFIX,
    }
    encoded = json.dumps(semantics, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def build_index(session, documents_jsonl, output_dir, batch_size=32):
    """Encode a small local knowledge base; keep document order in the artifact."""
    documents = _documents(Path(documents_jsonl))
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("index output already exists: " + str(output))
    vectors = session.encode(
        [item["text"] for item in documents], batch_size=batch_size
    )
    if vectors.shape != (len(documents), 512) or vectors.dtype != np.float32:
        raise ValueError("expected FP32 embedding matrix [N,512]")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".embedding-index-", dir=output.parent
    ) as folder:
        root = Path(folder)
        np.save(root / "vectors.npy", vectors, allow_pickle=False)
        with (root / "documents.jsonl").open("w", encoding="utf-8") as handle:
            for document in documents:
                handle.write(json.dumps(document, ensure_ascii=False) + "\n")
        manifest = {
            "format": 1,
            "count": len(documents),
            "dimensions": 512,
            "dtype": "float32",
            "normalization": "L2",
            "metric": "dot",
            "model_onnx_sha256": _model_hash(session),
            "embedding_fingerprint": _fingerprint(session),
            "model_revision": session.model_manifest["revision"],
            "sha256": {
                name: _digest(root / name)
                for name in ("vectors.npy", "documents.jsonl")
            },
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.rename(root, output)
    return manifest


def search_index(session, index_dir, query, top_k=10):
    top_k = _integer(top_k, "top_k")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    root = Path(index_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != 1 or manifest.get(
        "embedding_fingerprint"
    ) != _fingerprint(session):
        raise ValueError("index format or embedding model does not match")
    for name in ("vectors.npy", "documents.jsonl"):
        if _digest(root / name) != manifest["sha256"].get(name):
            raise ValueError("index checksum mismatch: " + name)
    documents = _documents(root / "documents.jsonl")
    vectors = np.load(root / "vectors.npy", mmap_mode="r", allow_pickle=False)
    if (
        vectors.shape != (len(documents), 512)
        or vectors.dtype != np.float32
        or len(documents) != manifest["count"]
    ):
        raise ValueError("index shape, dtype or document count does not match")
    query_vector = session.encode([query], kind="query")[0]
    matches = cosine_top_k(
        query_vector,
        vectors,
        min(top_k, len(documents)),
        ids=[d["id"] for d in documents],
    )
    return [dict(documents[index], score=score) for index, score in matches]
