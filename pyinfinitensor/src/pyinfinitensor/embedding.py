"""Chinese BGE embedding interface backed exclusively by InfiniTensor."""

from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Union
from contextlib import contextmanager
import time
import operator
import hashlib
import json
import re

import numpy as np

from . import backend
from .onnx import OnnxStub


BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("{} must be an integer".format(name))
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError("{} must be an integer".format(name)) from error


@contextmanager
def fp32_inference_policy() -> Iterator[None]:
    """Disable reduced-precision math for an accuracy-sensitive embedding call.

    Providers expose different public knobs.  Save and restore every knob we
    touch so creating an EmbeddingSession never changes the application's
    precision policy after a call returns.
    """
    try:
        import torch
    except ModuleNotFoundError:
        yield
        return
    saved = []

    def remember(path, value=None, change=False):
        owner = torch
        fields = path.split(".")
        try:
            for field in fields[:-1]:
                owner = getattr(owner, field)
            previous = getattr(owner, fields[-1])
        except AttributeError:
            return False
        if change:
            setattr(owner, fields[-1], value)
        saved.append(lambda: setattr(owner, fields[-1], previous))
        return True

    def override(path, value):
        return remember(path, value, change=True)

    try:
        # Preserve both the legacy policy (including "medium") and the raw
        # per-backend overrides changed by set_float32_matmul_precision.
        legacy = None
        if hasattr(torch, "get_float32_matmul_precision"):
            try:
                legacy = torch.get_float32_matmul_precision()
            except RuntimeError:
                # New per-backend policies may not have a single legacy
                # equivalent. In that case only touch the per-backend API.
                pass
        if legacy is not None:
            remember("backends.cuda.matmul.fp32_precision")
            remember("backends.mkldnn.matmul.fp32_precision")
            torch.set_float32_matmul_precision("highest")
            saved.append(lambda: torch.set_float32_matmul_precision(legacy))
        if (
            not override("backends.cuda.matmul.fp32_precision", "ieee")
            and legacy is None
        ):
            override("backends.cuda.matmul.allow_tf32", False)
        cudnn_legacy = None
        try:
            cudnn_legacy = torch.backends.cudnn.allow_tf32
        except (AttributeError, RuntimeError):
            pass
        if cudnn_legacy is not None:
            remember("backends.cudnn.conv.fp32_precision")
            remember("backends.cudnn.rnn.fp32_precision")
            override("backends.cudnn.allow_tf32", False)
        if override("backends.cudnn.conv.fp32_precision", "ieee"):
            override("backends.cudnn.rnn.fp32_precision", "ieee")
        for path in (
            "backends.mlu.matmul.allow_tf32",
            "backends.mlu.cnnl.allow_tf32",
            "npu.config.allow_internal_format",
            "npu.conv.allow_hf32",
            "npu.matmul.allow_hf32",
        ):
            override(path, False)
        yield
    finally:
        for restore in reversed(saved):
            restore()


def verify_model_package(model_dir: Path, model_path: Path):
    """Validate the frozen local package before constructing a device graph."""
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("model") != "BAAI/bge-small-zh-v1.5" or not re.fullmatch(
        r"[0-9a-f]{40}", manifest.get("revision", "")
    ):
        raise ValueError("expected a frozen BAAI/bge-small-zh-v1.5 model package")
    digests = manifest.get("sha256", {})
    if str(model_path.relative_to(model_dir)) not in digests:
        raise ValueError("ONNX model is not recorded in the package manifest")
    tokenizer_files = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.txt",
    )
    for name in tokenizer_files:
        if (model_dir / name).is_file() and name not in digests:
            raise ValueError("unrecorded tokenizer configuration: " + name)
    for name, expected in digests.items():
        path = model_dir / name
        if not path.resolve().is_relative_to(model_dir.resolve()):
            raise ValueError("model manifest path escapes the package directory")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError("model package checksum mismatch: " + name)
    return manifest


class EmbeddingSession:
    """Run a frozen BGE-small-zh ONNX package on one InfiniTensor device.

    Tokenization and final L2 normalization happen on CPU.  The encoder and
    CLS extraction are the ONNX graph executed by the selected provider.
    """

    def __init__(
        self,
        model_dir: Union[str, Path],
        device: str = "nvidia",
        device_id: int = 0,
        model_name: str = "model.onnx",
        max_batch_size: int = 32,
        max_length: int = 512,
    ):
        try:
            import onnx
            from transformers import AutoTokenizer
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "EmbeddingSession requires `pip install pyinfinitensor[embedding]`"
            ) from error
        self.model_dir = Path(model_dir)
        self.max_batch_size = _integer(max_batch_size, "max_batch_size")
        self.max_length = _integer(max_length, "max_length")
        if self.max_batch_size < 1 or self.max_batch_size > 32:
            raise ValueError("M1 certifies 1 <= max_batch_size <= 32")
        if self.max_length != 512:
            raise ValueError("BGE-small-zh M1 is certified at max_length=512")
        self.model_path = self._find_model(model_name)
        self.model_manifest = verify_model_package(self.model_dir, self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir), local_files_only=True
        )
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        self.stub = OnnxStub(
            onnx.load(str(self.model_path)),
            backend.runtime(device, _integer(device_id, "device_id")),
        )
        self.last_timing: Dict[str, float] = {}

    def _find_model(self, model_name: str) -> Path:
        candidates = [
            model_name,
            "model.onnx",
            "bge-small-zh-v1.5.onnx",
            "model_fp32.onnx",
        ]
        for candidate in dict.fromkeys(candidates):
            path = self.model_dir / candidate
            if path.is_file():
                return path
        raise FileNotFoundError("no ONNX model found in {}".format(self.model_dir))

    def encode(
        self,
        texts: Union[str, Sequence[str]],
        kind: str = "passage",
        batch_size: int = 32,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if kind not in {"query", "passage"}:
            raise ValueError('kind must be "query" or "passage"')
        batch_size = _integer(batch_size, "batch_size")
        if not 1 <= batch_size <= self.max_batch_size:
            raise ValueError(
                "batch_size must be in [1, {}]".format(self.max_batch_size)
            )
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("texts must be strings")
        if not texts:
            return np.empty((0, 512), dtype=np.float32)
        prepared = [
            BGE_QUERY_PREFIX + text if kind == "query" else text for text in texts
        ]
        vectors: List[np.ndarray] = []
        timings = {
            "tokenize_s": 0.0,
            "prepare_s": 0.0,
            "h2d_s": 0.0,
            "run_s": 0.0,
            "d2h_s": 0.0,
            "normalize_s": 0.0,
        }
        for start in range(0, len(prepared), int(batch_size)):
            chunk = prepared[start : start + int(batch_size)]
            started = time.perf_counter()
            encoded = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
                return_token_type_ids=True,
            )
            tokenized = time.perf_counter()
            inputs = {
                "input_ids": np.asarray(encoded["input_ids"], dtype=np.int64),
                "attention_mask": np.asarray(encoded["attention_mask"], dtype=np.int64),
                "token_type_ids": np.asarray(
                    encoded.get("token_type_ids", np.zeros_like(encoded["input_ids"])),
                    dtype=np.int64,
                ),
            }
            with fp32_inference_policy():
                outputs = self.stub.run_numpy(inputs)
            returned = time.perf_counter()
            cls = np.asarray(next(iter(outputs.values())), dtype=np.float32)
            if cls.ndim != 2 or cls.shape[0] != len(chunk) or cls.shape[1] != 512:
                raise RuntimeError(
                    "BGE model must return [batch, 512], got {}".format(cls.shape)
                )
            norms = np.linalg.norm(cls, axis=1, keepdims=True)
            if np.any(~np.isfinite(norms)) or np.any(norms == 0):
                raise RuntimeError("encoder returned a non-finite or zero embedding")
            vectors.append(cls / norms)
            normalized = time.perf_counter()
            timings["tokenize_s"] += tokenized - started
            for key in ("prepare_s", "h2d_s", "run_s", "d2h_s"):
                timings[key] += self.stub.last_timing.get(key, 0.0)
            timings["normalize_s"] += normalized - returned
        timings["total_s"] = sum(timings.values())
        self.last_timing = timings
        return np.ascontiguousarray(np.concatenate(vectors, axis=0), dtype=np.float32)


def cosine_top_k(query: np.ndarray, corpus: np.ndarray, top_k: int = 10, ids=None):
    """Return deterministic (index, score) pairs for normalized vectors."""
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    corpus = np.asarray(corpus, dtype=np.float32)
    if corpus.ndim != 2 or corpus.shape[1] != query.size:
        raise ValueError("query and corpus vector dimensions differ")
    if not 1 <= top_k <= len(corpus):
        raise ValueError("top_k must be in [1, corpus size]")
    if ids is not None and len(ids) != len(corpus):
        raise ValueError("ids and corpus must have the same length")
    scores = corpus @ query
    tie_break = np.arange(len(scores)) if ids is None else np.asarray(ids, dtype=str)
    order = np.lexsort((tie_break, -scores))[:top_k]
    return [(int(index), float(scores[index])) for index in order]
