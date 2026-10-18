"""Chinese BGE embedding interface backed exclusively by InfiniTensor."""

from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Union
from contextlib import contextmanager
import time

import numpy as np

from . import backend
from .onnx import OnnxStub


BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


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
    missing = object()
    saved = []
    cuda = getattr(torch, "backends", None)
    try:
        cuda_matmul = cuda.cuda.matmul
        saved.append((cuda_matmul, "allow_tf32", cuda_matmul.allow_tf32))
        cuda_matmul.allow_tf32 = False
        cudnn = cuda.cudnn
        saved.append((cudnn, "allow_tf32", cudnn.allow_tf32))
        cudnn.allow_tf32 = False
    except (AttributeError, RuntimeError):
        pass
    for owner_name, flag in (("mlu", "matmul"), ("mlu", "cnnl")):
        try:
            owner = getattr(torch.backends, owner_name)
            target = getattr(owner, flag)
            saved.append((target, "allow_tf32", target.allow_tf32))
            target.allow_tf32 = False
        except (AttributeError, RuntimeError):
            pass
    try:
        npu = torch.npu
    except (AttributeError, RuntimeError):
        npu = None
    if npu is not None:
        for owner, name in ((npu.config, "allow_internal_format"),
                            (npu.conv, "allow_hf32"), (npu.matmul, "allow_hf32")):
            try:
                saved.append((owner, name, getattr(owner, name, missing)))
                setattr(owner, name, False)
            except (AttributeError, RuntimeError):
                pass
    try:
        yield
    finally:
        for owner, name, value in reversed(saved):
            if value is missing:
                try:
                    delattr(owner, name)
                except AttributeError:
                    pass
            else:
                setattr(owner, name, value)


class EmbeddingSession:
    """Run a frozen BGE-small-zh ONNX package on one InfiniTensor device.

    Tokenization and final L2 normalization happen on CPU.  The encoder and
    CLS extraction are the ONNX graph executed by the selected provider.
    """

    def __init__(
        self, model_dir: Union[str, Path], device: str = "nvidia", device_id: int = 0,
        model_name: str = "model.onnx", max_batch_size: int = 32, max_length: int = 512,
        execution_mode: str = "normal", graph_capture_after: int = 4,
        memory_views: bool = False, fuse_linear_bias: bool = False,
    ):
        try:
            import onnx
            from transformers import AutoTokenizer
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "EmbeddingSession requires `pip install pyinfinitensor[embedding]`"
            ) from error
        self.model_dir = Path(model_dir)
        self.max_batch_size = int(max_batch_size)
        self.max_length = int(max_length)
        if self.max_batch_size < 1 or self.max_batch_size > 32:
            raise ValueError("M1 certifies 1 <= max_batch_size <= 32")
        if self.max_length != 512:
            raise ValueError("BGE-small-zh M1 is certified at max_length=512")
        self.model_path = self._find_model(model_name)
        from .model_package import package_identity
        self.package_identity = package_identity(self.model_dir, self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
        self.stub = OnnxStub(onnx.load(str(self.model_path)),
                             backend.runtime(device, int(device_id), graph_cache_capacity=1),
                             execution_mode=execution_mode,
                             graph_capture_after=graph_capture_after,
                             memory_views=memory_views,
                             fuse_linear_bias=fuse_linear_bias)
        # Reserve the certified upper-bound activation pool once.  Growing a
        # device allocation while the previous executable state remains valid
        # temporarily needs both pools, which is fragile on shared GPUs.  All
        # later dynamic shapes reuse this capacity; weights remain single-copy.
        certified_shape = {
            name: (self.max_batch_size, self.max_length)
            for name in ("input_ids", "attention_mask", "token_type_ids")
        }
        self.stub.prepare(certified_shape)
        self.last_timing: Dict[str, float] = {}

    def execution_stats(self) -> Dict[str, object]:
        return self.stub.execution_stats()

    def _find_model(self, model_name: str) -> Path:
        candidates = [model_name, "model.onnx", "bge-small-zh-v1.5.onnx", "model_fp32.onnx"]
        for candidate in dict.fromkeys(candidates):
            path = self.model_dir / candidate
            if path.is_file():
                return path
        raise FileNotFoundError("no ONNX model found in {}".format(self.model_dir))

    def encode(
        self, texts: Union[str, Sequence[str]], kind: str = "passage", batch_size: int = 32
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if kind not in {"query", "passage"}:
            raise ValueError('kind must be "query" or "passage"')
        if not 1 <= int(batch_size) <= self.max_batch_size:
            raise ValueError("batch_size must be in [1, {}]".format(self.max_batch_size))
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("texts must be strings")
        if not texts:
            return np.empty((0, 512), dtype=np.float32)
        prepared = [BGE_QUERY_PREFIX + text if kind == "query" else text for text in texts]
        vectors: List[np.ndarray] = []
        timings = {"tokenize_s": 0.0, "prepare_s": 0.0, "h2d_s": 0.0, "run_s": 0.0, "d2h_s": 0.0, "normalize_s": 0.0}
        for start in range(0, len(prepared), int(batch_size)):
            chunk = prepared[start : start + int(batch_size)]
            started = time.perf_counter()
            encoded = self.tokenizer(
                chunk, padding=True, truncation=True, max_length=self.max_length,
                return_tensors="np", return_token_type_ids=True,
            )
            tokenized = time.perf_counter()
            inputs = {
                "input_ids": np.asarray(encoded["input_ids"], dtype=np.int64),
                "attention_mask": np.asarray(encoded["attention_mask"], dtype=np.int64),
                "token_type_ids": np.asarray(
                    encoded.get("token_type_ids", np.zeros_like(encoded["input_ids"])), dtype=np.int64
                ),
            }
            with fp32_inference_policy():
                outputs = self.stub.run_numpy(inputs)
            returned = time.perf_counter()
            cls = np.asarray(next(iter(outputs.values())), dtype=np.float32)
            if cls.ndim != 2 or cls.shape[0] != len(chunk) or cls.shape[1] != 512:
                raise RuntimeError("BGE model must return [batch, 512], got {}".format(cls.shape))
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
