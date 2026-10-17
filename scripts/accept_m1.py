#!/usr/bin/env python3
"""Reproducible M1 acceptance runner for dynamic BGE-small-zh embedding.

The runner deliberately has no inference fallback.  Each engine is constructed
explicitly and a missing provider, dataset, or evidence file is a failure.
"""
import argparse
import contextlib
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import resource
import re
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

REVISION = "8731a845f1bf500a4f111cf1070785c793d10e64"
CORPUS_FILE = "corpus-00000-of-00001-8afe7b7a7eca49e3.parquet"
QUERIES_FILE = "queries-00000-of-00001-930bf3b805a80dd9.parquet"
CASES = [(1, 32), (1, 128), (1, 512), (8, 128), (32, 128)]
STABILITY_SHAPES = [(1, 1), (1, 17), (3, 511), (8, 128), (32, 512), (1, 17)]
PROCESS_STARTED = time.perf_counter()
NVIDIA_PROCESS = {}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def device_env(device: str) -> Dict[str, Any]:
    answer = {"hostname": platform.node(), "platform": platform.platform(),
              "python": sys.version, "pid": os.getpid(), "device": device,
              "started_unix_s": time.time()}
    try:
        import torch
        answer["torch"] = torch.__version__
        answer["torch_abi"] = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    except Exception as error:
        answer["torch_error"] = repr(error)
    for command in (("nvidia-smi", "--query-gpu=index,name,memory.used,memory.free,utilization.gpu", "--format=csv,noheader"),
                    ("cnmon", "all"), ("npu-smi", "info")):
        try:
            answer[command[0]] = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, timeout=15)
        except (OSError, subprocess.SubprocessError):
            pass
    return answer


def model_metadata(model_dir: Path) -> Dict[str, Any]:
    manifest = model_dir / "manifest.json"
    model = model_dir / "model.onnx"
    if not model.is_file():
        raise FileNotFoundError("M1 model package lacks {}".format(model))
    raw = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
    actual_hash = sha256(model)
    expected_hash = raw.get("sha256", {}).get("model.onnx")
    if expected_hash and actual_hash != expected_hash:
        raise RuntimeError("model.onnx hash does not match manifest")
    export_validation = raw.get("pytorch_ort_validation")
    if not isinstance(export_validation, str) or not export_validation.lower().startswith("passed"):
        raise RuntimeError("model package lacks passing PyTorch-to-ORT export validation")
    return {"model_onnx_sha256": actual_hash, "model": raw.get("model"),
            "revision": raw.get("revision"), "opset": raw.get("opset"),
            "artifact_hashes": raw.get("sha256"), "node_count": len(raw.get("nodes", [])),
            "pytorch_ort_validation": raw.get("pytorch_ort_validation")}


def source_evidence() -> Dict[str, Any]:
    """Hash the executable Python acceptance path used by this process."""
    files = {"accept_m1": Path(__file__).resolve()}
    result: Dict[str, Any] = {}
    try:
        import pyinfinitensor
        from pyinfinitensor import embedding, onnx
        files.update(pyinfinitensor_init=Path(pyinfinitensor.__file__).resolve(),
                     embedding=Path(embedding.__file__).resolve(),
                     onnx=Path(onnx.__file__).resolve())
    except BaseException as error:
        result["import_error"] = "{}: {}".format(type(error).__name__, error)
    result.update({name: {"path": str(path), "sha256": sha256(path)}
                   for name, path in files.items()})
    from finalize_m1_acceptance import ENGINE_PARTS, DELIVERY_PARTS, source_manifest
    result["engine_tree_sha256"] = source_manifest(
        Path(__file__).resolve().parents[1], ENGINE_PARTS)["tree_sha256"]
    result["delivery_tree_sha256"] = source_manifest(
        Path(__file__).resolve().parents[1], DELIVERY_PARTS)["tree_sha256"]
    return result


def make_inputs(batch: int, sequence: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    inputs = {"input_ids": rng.integers(0, 1000, (batch, sequence), dtype=np.int64),
              "attention_mask": np.ones((batch, sequence), dtype=np.int64),
              "token_type_ids": np.zeros((batch, sequence), dtype=np.int64)}
    for row in range(batch):
        valid = int(rng.integers(1, sequence + 1))
        inputs["attention_mask"][row, valid:] = 0
        inputs["input_ids"][row, valid:] = 0
    return inputs


def normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return value / np.linalg.norm(value, axis=-1, keepdims=True)


def import_engine(args):
    from pyinfinitensor.embedding import EmbeddingSession
    return EmbeddingSession(args.model_dir, device=args.device, device_id=args.device_id,
                            execution_mode=getattr(args, "execution_mode", "auto"),
                            graph_capture_after=getattr(args, "graph_capture_after", 4))


def torch_device(device: str, index: int) -> str:
    return {"nvidia": "cuda", "cambricon": "mlu", "ascend": "npu"}[device] + ":{}".format(index)


@contextlib.contextmanager
def torch_reference(model_dir: Path, device: str, index: int):
    import torch
    from transformers import AutoModel
    from pyinfinitensor.embedding import fp32_inference_policy
    target = torch_device(device, index)
    with fp32_inference_policy():
        model = AutoModel.from_pretrained(model_dir, local_files_only=True,
                                          attn_implementation="eager").eval().to(target)
        try:
            yield model, target
        finally:
            del model
            allocator = {"nvidia": getattr(torch, "cuda", None),
                         "cambricon": getattr(torch, "mlu", None),
                         "ascend": getattr(torch, "npu", None)}[device]
            if allocator is not None and hasattr(allocator, "empty_cache"):
                allocator.empty_cache()


def torch_forward(model, target: str, inputs: Dict[str, np.ndarray]) -> np.ndarray:
    import torch
    with torch.inference_mode():
        values = {name: torch.from_numpy(value).to(target) for name, value in inputs.items()}
        output = model(**values).last_hidden_state[:, 0, :]
        return output.cpu().numpy()


def check_preflight(args, output: Path) -> Dict[str, Any]:
    session = import_engine(args)
    shapes = {"input_ids": (1, 1), "attention_mask": (1, 1), "token_type_ids": (1, 1)}
    report = session.stub.check_capabilities(shapes)
    report["model"] = model_metadata(Path(args.model_dir))
    # This actual zero-input dispatch proves the selected provider contains the
    # compiled bridges, not merely that the frontend can import the graph.
    values = {name: np.zeros(shape, dtype=np.int64) for name, shape in shapes.items()}
    result = session.stub.run_numpy(values)["embeddings"]
    if result.shape != (1, 512) or not np.isfinite(result).all():
        raise RuntimeError("preflight execution returned invalid BGE embedding {}".format(result.shape))
    report["actual_device_dispatch"] = True
    write_json(output / "preflight.json", report)
    return report


def check_accuracy(args, output: Path) -> Dict[str, Any]:
    rng = np.random.default_rng(315)
    shapes = [(1, 17), (7, 83), (3, 511), (16, 128), (1, 17), (1, 1), (32, 1)]
    shapes += [(b, s) for b in (1, 3, 8, 17, 32) for s in (2, 17, 64, 127, 255, 511, 512)]
    shapes += [(int(rng.integers(1, 33)), int(rng.integers(1, 513))) for _ in range(args.random_shapes)]
    # Keep reference and engine allocations disjoint.  Shared acceptance hosts
    # may have only a few GiB free even though either engine fits on its own.
    # Inputs and CLS outputs stay on CPU and are small enough to retain.
    prepared = [make_inputs(batch, sequence, rng) for batch, sequence in shapes]
    expected_values = []
    with torch_reference(Path(args.model_dir), args.device, args.device_id) as (reference, target):
        for inputs in prepared:
            expected_values.append(torch_forward(reference, target, inputs))

    session = import_engine(args)
    uploads, upload_bytes, addresses = (session.stub.weight_upload_count,
                                         session.stub.weight_upload_bytes,
                                         session.stub.weight_addresses)
    negative_cases = []
    base = prepared[0]
    def rejected(label, callback, error_types):
        before = session.stub.memory_stats()
        try:
            callback()
        except error_types as error:
            negative_cases.append({"case": label, "error": "{}: {}".format(type(error).__name__, error)})
        else:
            raise AssertionError("{} was not rejected".format(label))
        after = session.stub.memory_stats()
        if before["allocation_generation"] != after["allocation_generation"]:
            raise AssertionError("{} reached device allocation".format(label))
        if uploads != session.stub.weight_upload_count or addresses != session.stub.weight_addresses:
            raise AssertionError("{} changed resident weights".format(label))
    rejected("input names", lambda: session.stub.run_numpy({name: value for name, value in base.items()
                                                              if name != "token_type_ids"}), ValueError)
    bad_dtype = dict(base); bad_dtype["input_ids"] = base["input_ids"].astype(np.float32)
    rejected("input dtype", lambda: session.stub.run_numpy(bad_dtype), TypeError)
    bad_rank = dict(base); bad_rank["input_ids"] = base["input_ids"][..., None]
    rejected("input rank", lambda: session.stub.run_numpy(bad_rank), ValueError)
    bad_shared = dict(base); bad_shared["attention_mask"] = np.ones((1, 18), np.int64)
    rejected("shared dimensions", lambda: session.stub.run_numpy(bad_shared), ValueError)
    rejected("non-integer shape", lambda: session.stub.prepare({name: (1.5, 17) for name in base}), ValueError)
    rejected("non-positive shape", lambda: session.stub.prepare({name: (0, 17) for name in base}), ValueError)
    items, prior = [], None
    from pyinfinitensor.embedding import fp32_inference_policy
    for ordinal, ((batch, sequence), inputs, expected) in enumerate(zip(shapes, prepared, expected_values)):
        with fp32_inference_policy():
            actual = session.stub.run_numpy(inputs)["embeddings"]
        if prior is not None:
            np.testing.assert_array_equal(prior[0], prior[1])
        prior = (actual, actual.copy())
        got, want = normalize(actual), normalize(expected)
        np.testing.assert_allclose(got, want, atol=1e-4, rtol=1e-3)
        cosine = np.sum(got * want, axis=1)
        if float(cosine.min()) < 0.99999:
            raise AssertionError("cosine below threshold at {}x{}".format(batch, sequence))
        if uploads != session.stub.weight_upload_count or upload_bytes != session.stub.weight_upload_bytes or addresses != session.stub.weight_addresses:
            raise AssertionError("initializer residency changed at request {}".format(ordinal))
        items.append({"shape": [batch, sequence], "max_abs_error": float(np.abs(got - want).max()),
                      "min_cosine": float(cosine.min())})
    # Text API acceptance: batching, prefix, empty list, truncation, and
    # single-versus-mixed padding behavior.
    texts = ["中文语义检索", "InfiniTensor supports embedding！", "", "标点，。！？", "长文本" * 600]
    many = texts * 9
    encoded = session.encode(many, kind="passage", batch_size=32)
    if encoded.shape != (len(many), 512) or encoded.dtype != np.float32 or not np.isfinite(encoded).all():
        raise AssertionError("text batching returned invalid shape")
    norm_error = float(np.max(np.abs(np.linalg.norm(encoded, axis=1) - 1.0)))
    if norm_error > 1e-5:
        raise AssertionError("normalized vector norm error exceeds 1e-5")
    if session.encode([], kind="passage").shape != (0, 512):
        raise AssertionError("empty text list is not stable")
    single = session.encode([texts[0]])
    mixed = session.encode([texts[0], texts[-1]])[:1]
    np.testing.assert_allclose(single, mixed, atol=1e-4, rtol=1e-3)
    query = session.encode([texts[0]], kind="query")
    if query.shape != (1, 512) or not np.isfinite(query).all():
        raise AssertionError("query-prefix encoding returned invalid output")
    for offset in range(len(texts)):
        np.testing.assert_allclose(encoded[offset], encoded[offset + len(texts)], atol=1e-6, rtol=1e-6)
    report = {"status": "PASS", "requests": len(items), "items": items,
              "min_cosine": min(item["min_cosine"] for item in items),
              "max_abs_error": max(item["max_abs_error"] for item in items),
              "weight_upload_count": uploads, "weight_upload_bytes": upload_bytes,
              "weight_addresses_stable": True, "text_batch_rows": len(many),
              "norm_max_error": norm_error, "query_prefix_executed": True,
              "negative_cases": negative_cases, "execution": session.execution_stats()}
    write_json(output / "accuracy.json", report)
    return report


def download_dataset(dataset_dir: Path) -> None:
    import urllib.request
    dataset_dir.mkdir(parents=True, exist_ok=True)
    bases = [os.environ.get("M1_DATASET_BASE_URL", "https://huggingface.co"), "https://hf-mirror.com"]
    for name in (CORPUS_FILE, QUERIES_FILE):
        path = dataset_dir / name
        if not path.is_file():
            errors = []
            for base in dict.fromkeys(bases):
                url = "{}/datasets/C-MTEB/T2Retrieval/resolve/{}/data/{}?download=true".format(base.rstrip("/"), REVISION, name)
                try:
                    urllib.request.urlretrieve(url, path)
                    break
                except OSError as error:
                    errors.append("{}: {}".format(base, error))
            else:
                raise OSError("unable to download fixed T2Retrieval data: {}".format("; ".join(errors)))


def retrieval_data(dataset_dir: Path, output: Path) -> Tuple[List[str], List[str], List[str], List[str], str]:
    try:
        import pandas as pd
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("retrieval acceptance requires pandas and pyarrow") from error
    corpus = pd.read_parquet(dataset_dir / CORPUS_FILE).astype({"id": str, "text": str}).sort_values("id").head(1000)
    queries = pd.read_parquet(dataset_dir / QUERIES_FILE).astype({"id": str, "text": str}).sort_values("id").head(100)
    selected = {"dataset_revision": REVISION, "rule": "string id ascending; corpus first 1000; queries first 100",
                "corpus": corpus.to_dict("records"), "queries": queries.to_dict("records"),
                "source_file_sha256": {CORPUS_FILE: sha256(dataset_dir / CORPUS_FILE),
                                       QUERIES_FILE: sha256(dataset_dir / QUERIES_FILE)}}
    canonical = json.dumps(selected, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    selected["selection_sha256"] = hashlib.sha256(canonical).hexdigest()
    write_json(output / "retrieval-selection.json", selected)
    return (queries.id.tolist(), queries.text.tolist(), corpus.id.tolist(), corpus.text.tolist(),
            selected["selection_sha256"])


def stable_top10(query: np.ndarray, corpus: np.ndarray, ids: Sequence[str]) -> List[List[str]]:
    score = corpus @ query.T
    answer = []
    ids = np.asarray(ids)
    for column in range(score.shape[1]):
        order = np.lexsort((ids, -score[:, column]))[:10]
        answer.append(ids[order].tolist())
    return answer


def check_retrieval(args, output: Path) -> Dict[str, Any]:
    dataset_dir = Path(args.dataset_dir)
    if args.download_dataset:
        download_dataset(dataset_dir)
    q_ids, queries, d_ids, documents, selection_hash = retrieval_data(dataset_dir, output)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    with torch_reference(Path(args.model_dir), args.device, args.device_id) as (reference, target):
        def reference_encode(texts, kind):
            prefix = "为这个句子生成表示以用于检索相关文章：" if kind == "query" else ""
            pieces = []
            for start in range(0, len(texts), 32):
                encoded = tokenizer([prefix + text for text in texts[start:start + 32]], padding=True,
                                    truncation=True, max_length=512, return_tensors="np", return_token_type_ids=True)
                inputs = {"input_ids": np.asarray(encoded["input_ids"], np.int64),
                          "attention_mask": np.asarray(encoded["attention_mask"], np.int64),
                          "token_type_ids": np.asarray(encoded.get("token_type_ids", np.zeros_like(encoded["input_ids"])), np.int64)}
                pieces.append(normalize(torch_forward(reference, target, inputs)))
            return np.concatenate(pieces)
        wanted_docs, wanted_queries = reference_encode(documents, "passage"), reference_encode(queries, "query")
    session = import_engine(args)
    actual_docs = session.encode(documents, kind="passage", batch_size=32)
    actual_queries = session.encode(queries, kind="query", batch_size=32)
    np.testing.assert_allclose(actual_docs, wanted_docs, atol=1e-4, rtol=1e-3)
    np.testing.assert_allclose(actual_queries, wanted_queries, atol=1e-4, rtol=1e-3)
    actual_top = stable_top10(actual_queries, actual_docs, d_ids)
    reference_top = stable_top10(wanted_queries, wanted_docs, d_ids)
    overlaps = [len(set(left) & set(right)) / 10.0 for left, right in zip(actual_top, reference_top)]
    vector_cosine = np.concatenate((np.sum(actual_docs * wanted_docs, axis=1),
                                    np.sum(actual_queries * wanted_queries, axis=1)))
    report = {"status": "PASS" if float(np.mean(overlaps)) >= .99 else "FAIL", "dataset_revision": REVISION,
              "queries": len(q_ids), "documents": len(d_ids), "mean_top10_overlap": float(np.mean(overlaps)),
              "min_top10_overlap": float(np.min(overlaps)), "query_ids_sha256": hashlib.sha256("\n".join(q_ids).encode()).hexdigest(),
              "document_ids_sha256": hashlib.sha256("\n".join(d_ids).encode()).hexdigest(),
              "selection_sha256": selection_hash,
              "max_vector_abs_error": float(max(np.abs(actual_docs - wanted_docs).max(),
                                                  np.abs(actual_queries - wanted_queries).max())),
              "min_vector_cosine": float(vector_cosine.min()),
              "actual_top10": actual_top, "reference_top10": reference_top,
              "execution": session.execution_stats()}
    if report["status"] != "PASS":
        raise AssertionError("Top-10 overlap below 99%: {}".format(report["mean_top10_overlap"]))
    write_json(output / "retrieval.json", report)
    return report


def current_rss_bytes() -> int:
    with open("/proc/self/statm", encoding="utf-8") as handle:
        return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def nvidia_process_inventory():
    rows = subprocess.check_output(("nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                    "--format=csv,noheader,nounits"), text=True, timeout=10)
    result = {}
    for row in rows.splitlines():
        uuid, pid, memory = (part.strip() for part in row.split(","))
        result[(uuid, int(pid))] = int(memory) * 1024**2
    return result


def resolve_nvidia_process(device_id):
    """Resolve a container PID without attributing whole-device memory to us.

    Used only before stability timing, never inside cold-start benchmarks.
    Two small held allocations identify our NVML process through matching
    memory deltas when the host PID namespace is hidden. Ambiguity is an error.
    """
    import torch
    before_init = nvidia_process_inventory()
    was_initialized = torch.cuda.is_initialized()
    torch.cuda.init()
    context = torch.empty(1, dtype=torch.uint8, device="cuda:{}".format(device_id))
    torch.cuda.synchronize(device_id)
    uuid = str(torch.cuda.get_device_properties(device_id).uuid)
    if not uuid.startswith("GPU-"):
        uuid = "GPU-" + uuid
    after_init = nvidia_process_inventory()
    if (uuid, os.getpid()) in after_init:
        answer = {"uuid": uuid, "pid": os.getpid(), "method": "direct PID"}
    else:
        new = [key for key in after_init if key[0] == uuid and key not in before_init]
        if not was_initialized and len(new) == 1:
            answer = {"uuid": uuid, "pid": new[0][1], "method": "unique CUDA context creation"}
        else:
            before = nvidia_process_inventory()
            first = torch.empty(16 * 1024**2, dtype=torch.uint8, device="cuda:{}".format(device_id))
            torch.cuda.synchronize(device_id)
            middle = nvidia_process_inventory()
            second = torch.empty(32 * 1024**2, dtype=torch.uint8, device="cuda:{}".format(device_id))
            torch.cuda.synchronize(device_id)
            after = nvidia_process_inventory()
            candidates = [key for key in before.keys() & middle.keys() & after.keys()
                          if key[0] == uuid
                          and abs((middle[key] - before[key]) - 16 * 1024**2) <= 4 * 1024**2
                          and abs((after[key] - middle[key]) - 32 * 1024**2) <= 4 * 1024**2]
            del first, second
            torch.cuda.empty_cache()
            if len(candidates) != 1:
                raise RuntimeError("cannot uniquely attribute NVML memory to this container process")
            answer = {"uuid": uuid, "pid": candidates[0][1], "method": "two controlled allocation deltas",
                      "before": before[candidates[0]], "middle": middle[candidates[0]], "after": after[candidates[0]]}
    NVIDIA_PROCESS.update(answer)
    del context
    torch.cuda.empty_cache()
    return answer


def resource_sample(device: str, session=None) -> Dict[str, Any]:
    sample = {"unix_s": time.time(), "rss_bytes": current_rss_bytes(),
              "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    if session is not None:
        sample["infinitensor"] = session.stub.memory_stats()
    try:
        import torch
        if device == "nvidia":
            sample.update(aten_allocated_bytes=int(torch.cuda.memory_allocated()), aten_reserved_bytes=int(torch.cuda.memory_reserved()))
        elif device == "cambricon":
            sample.update(aten_allocated_bytes=int(torch.mlu.memory_allocated()), aten_reserved_bytes=int(torch.mlu.memory_reserved()))
        elif device == "ascend":
            sample.update(aten_allocated_bytes=int(torch.npu.memory_allocated()), aten_reserved_bytes=int(torch.npu.memory_reserved()))
    except Exception as error:
        sample["aten_memory_error"] = repr(error)
    stats = sample.get("infinitensor", {})
    if stats:
        sample["managed_device_bytes"] = int(
            stats.get("weight_bytes", 0) + stats.get("activation_bytes", 0)
            + stats.get("heap_bytes", 0) + sample.get("aten_reserved_bytes", 0))
    try:
        if device == "nvidia":
            rows = subprocess.check_output(
                ("nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                 "--format=csv,noheader,nounits"), text=True,
                stderr=subprocess.STDOUT, timeout=10)
            sample["external_process_inventory"] = rows
            sample["external_device_load"] = subprocess.check_output(
                ("nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu",
                 "--format=csv,noheader,nounits"), text=True, timeout=10)
            for row in rows.splitlines():
                uuid, pid, memory = (part.strip() for part in row.split(","))
                if (int(pid) == NVIDIA_PROCESS.get("pid", os.getpid())
                        and (not NVIDIA_PROCESS or uuid == NVIDIA_PROCESS["uuid"])):
                    sample["process_device_memory_bytes"] = int(memory) * 1024 * 1024
                    sample["process_device_memory_identity"] = dict(NVIDIA_PROCESS)
                    break
        elif device == "ascend":
            rows = subprocess.check_output(("npu-smi", "info"), text=True,
                                           stderr=subprocess.STDOUT, timeout=15)
            sample["external_device_load"] = rows
            for row in rows.splitlines():
                columns = [part.strip() for part in row.split("|")]
                if len(columns) >= 6 and columns[2] == str(os.getpid()):
                    sample["process_device_memory_bytes"] = int(columns[4]) * 1024 * 1024
                    break
        elif device == "cambricon":
            rows = subprocess.check_output(
                ("cnmon", "all"), text=True, stderr=subprocess.STDOUT, timeout=15)
            sample["external_device_load"] = rows
            for match in re.finditer(r"^\|\s*\d+\s+\S+\s+(\d+)\s+.*?\s+(\d+)\s+MiB\s*\|", rows, re.M):
                if int(match.group(1)) == os.getpid():
                    sample["process_device_memory_bytes"] = int(match.group(2)) * 1024 * 1024
                    sample["process_device_memory_source"] = "cnmon all (container PID namespace)"
                    break
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        sample["process_device_memory_error"] = repr(error)
    return sample


def stability_resource_evidence(samples, elapsed_s):
    """Use elapsed-time windows, never substitute sample counts for time."""
    if len(samples) < 4:
        raise AssertionError("insufficient resource samples")
    def growth(field, label, required=True):
        available = [item for item in samples if field in item]
        if required and len(available) < len(samples) * .95:
            raise AssertionError("{} evidence missing from {}/{} samples".format(
                label, len(samples) - len(available), len(samples)))
        first = [item for item in available if item["elapsed_s"] <= 1800]
        last = [item for item in available if item["elapsed_s"] >= elapsed_s - 1800]
        trend = [item for item in available if item["elapsed_s"] >= elapsed_s - 3600]
        for name, window, duration in (("first30m", first, 1800), ("last30m", last, 1800),
                                       ("last60m", trend, 3600)):
            if len(window) < 4 or window[-1]["elapsed_s"] - window[0]["elapsed_s"] < duration * .95:
                raise AssertionError("{} {} time coverage is insufficient".format(label, name))
        first_value = float(np.median([item[field] for item in first]))
        last_value = float(np.median([item[field] for item in last]))
        x = np.asarray([item["elapsed_s"] for item in trend], dtype=np.float64)
        x -= x.mean()
        y = np.asarray([item[field] for item in trend], dtype=np.float64)
        y -= y.mean()
        slope = float(np.dot(x, y) / np.dot(x, x))
        allowed = max(64 * 1024 * 1024, .05 * first_value)
        if last_value - first_value > allowed or slope > 1024 * 1024 / 60:
            raise AssertionError("{} did not stabilize: first={}, last={}, slope={}".format(
                label, first_value, last_value, slope))
        return {"samples": len(available), "first_window_median_bytes": first_value,
                "last_window_median_bytes": last_value, "allowed_growth_bytes": allowed,
                "last_60min_slope_bytes_s": slope, "trend_method": "ordinary_least_squares",
                "first_window_samples": len(first), "last_window_samples": len(last),
                "trend_window_samples": len(trend)}
    evidence = {"rss": growth("rss_bytes", "RSS"),
                "aten_reserved": growth("aten_reserved_bytes", "ATen reserved memory")}
    evidence["process_device_memory"] = growth("process_device_memory_bytes", "process device memory")
    for field, name in (("activation_bytes", "infinitensor_activation"),
                        ("heap_bytes", "infinitensor_heap")):
        values = {item["infinitensor"][field] for item in samples}
        if len(values) != 1:
            raise AssertionError("{} allocation changed at the fixed check shape: {}".format(name, values))
        evidence[name + "_bytes"] = next(iter(values))
    evidence["managed_device"] = growth("managed_device_bytes", "managed device memory")
    evidence["resource_check_shape"] = [8, 128]
    return evidence


def check_stability(args, output: Path) -> Dict[str, Any]:
    from pyinfinitensor.embedding import fp32_inference_policy
    if args.device == "nvidia":
        write_json(output / "nvml-process-identity.json", resolve_nvidia_process(args.device_id))
    session = import_engine(args)
    rng = np.random.default_rng(315)
    uploads, upload_bytes, addresses = (session.stub.weight_upload_count,
        session.stub.weight_upload_bytes, session.stub.weight_addresses)
    count, samples, previous, comparisons = 0, [], None, []
    input_digest = hashlib.sha256()
    with fp32_inference_policy(), torch_reference(
            Path(args.model_dir), args.device, args.device_id) as (reference, target):
        # Warm both engines at all shapes before starting the continuous clock.
        for b, s in STABILITY_SHAPES:
            values = make_inputs(b, s, rng)
            for _ in range(args.stability_repeat):
                session.stub.run_numpy(values)
            torch_forward(reference, target, values)
        started = time.monotonic()
        next_sample, last_check = started, -1
        with (output / "stability-resources.jsonl").open("w", encoding="utf-8", buffering=1) as resource_log:
            while count < args.stability_requests or time.monotonic() - started < args.stability_seconds:
                index = (count // args.stability_repeat + count // 100) % len(STABILITY_SHAPES)
                b, s = STABILITY_SHAPES[index]
                values = make_inputs(b, s, rng)
                input_digest.update(values["input_ids"].tobytes())
                actual = session.stub.run_numpy(values)["embeddings"]
                if not np.isfinite(actual).all():
                    raise AssertionError("non-finite output at request {}".format(count))
                if previous is not None:
                    np.testing.assert_array_equal(previous[0], previous[1])
                previous = (actual, actual.copy())
                execution = session.execution_stats()
                phase_ready = (execution["mode"] == "normal" or execution["last_execution"] == "replay")
                if count // 100 > last_check and phase_ready:
                    want = normalize(torch_forward(reference, target, values))
                    got = normalize(actual)
                    np.testing.assert_allclose(got, want, atol=1e-4, rtol=1e-3)
                    cosine = float(np.sum(got * want, axis=1).min())
                    norm_error = float(np.max(np.abs(np.linalg.norm(got, axis=1) - 1)))
                    if cosine < .99999 or norm_error > 1e-5:
                        raise AssertionError("stability reference comparison failed at {}".format(count))
                    comparisons.append({"request": count, "shape": [b, s], "min_cosine": cosine,
                                        "max_abs_error": float(np.abs(got - want).max())})
                    last_check = count // 100
                if (uploads != session.stub.weight_upload_count or upload_bytes != session.stub.weight_upload_bytes
                        or addresses != session.stub.weight_addresses):
                    raise AssertionError("weight residency changed during stability")
                count += 1
                # Observe a single identical allocation shape for all resource samples.
                if (b, s) == (8, 128) and phase_ready and time.monotonic() >= next_sample:
                    sample = resource_sample(args.device, session)
                    sample.update(elapsed_s=time.monotonic() - started, device=args.device, execution=execution,
                                  checked_shape=[b, s], requests=count)
                    samples.append(sample)
                    resource_log.write(json.dumps(sample, sort_keys=True) + "\n")
                    write_json(output / "stability-progress.json", {"method_version": 2,
                        "requests": count, "elapsed_s": time.monotonic() - started, "execution": execution,
                        "reference_checks": len(comparisons), "samples": samples[-120:]})
                    next_sample = time.monotonic() + 5
    elapsed_s = time.monotonic() - started
    full = count >= 10000 and elapsed_s >= 7200
    evidence = stability_resource_evidence(samples, elapsed_s) if full else {}
    report = {"status": "PASS" if full else "CALIBRATED", "method_version": 2,
              "requests": count, "elapsed_s": elapsed_s, "seed": 315,
              "input_stream_sha256": input_digest.hexdigest(), "warmup_completed": True,
              "reference_engine": "pytorch-" + args.device, "reference_checks": len(comparisons),
              "comparisons": comparisons, "weight_upload_count": uploads,
              "weight_upload_bytes": upload_bytes, "weight_addresses_stable": True,
              "resource_evidence": evidence, "samples": samples,
              "shape_repeat": args.stability_repeat, "execution": session.execution_stats()}
    write_json(output / "stability.json", report)
    return report


def percentile(values: Sequence[float], point: int) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), point))


def benchmark_infinitensor(session, texts, batch: int, warmup: int, runs: int) -> Dict[str, Any]:
    requests = benchmark_requests(texts)
    first_started = time.perf_counter()
    session.encode(next(requests), batch_size=batch)
    first_request = time.perf_counter() - first_started
    for _ in range(warmup):
        session.encode(next(requests), batch_size=batch)
    samples, segments = [], {key: [] for key in session.last_timing}
    for _ in range(runs):
        now = time.perf_counter(); session.encode(next(requests), batch_size=batch); samples.append(time.perf_counter() - now)
        for name, value in session.last_timing.items(): segments.setdefault(name, []).append(value)
    return {"samples_s": samples, "first_request_s": first_request,
            "segments_p50_s": {name: percentile(values, 50) for name, values in segments.items()}}


def sync_device(device: str) -> None:
    import torch
    if device == "nvidia": torch.cuda.synchronize()
    elif device == "cambricon": torch.mlu.synchronize()
    elif device == "ascend": torch.npu.synchronize()


def token_inputs(session, texts: Sequence[str]) -> Dict[str, np.ndarray]:
    encoded = session.tokenizer(texts, padding=True, truncation=True, max_length=512,
                                return_tensors="np", return_token_type_ids=True)
    return {"input_ids": np.asarray(encoded["input_ids"], np.int64),
            "attention_mask": np.asarray(encoded["attention_mask"], np.int64),
            "token_type_ids": np.asarray(encoded.get("token_type_ids", np.zeros_like(encoded["input_ids"])), np.int64)}


def benchmark_requests(texts):
    return itertools.cycle(texts) if texts and isinstance(texts[0], list) else itertools.repeat(texts)


def benchmark_torch(session, model_dir: Path, device: str, device_id: int, texts, warmup: int, runs: int) -> Dict[str, Any]:
    import torch
    load_started = time.perf_counter()
    requests = benchmark_requests(texts)
    with torch_reference(model_dir, device, device_id) as (model, target):
        model_load_s = time.perf_counter() - load_started
        def once():
            begun = time.perf_counter(); inputs = token_inputs(session, next(requests)); tokenized = time.perf_counter()
            values = {name: torch.from_numpy(value).to(target) for name, value in inputs.items()}
            sync_device(device); copied_in = time.perf_counter()
            with torch.inference_mode():
                device_output = model(**values).last_hidden_state[:, 0, :]
            sync_device(device); executed = time.perf_counter()
            output = device_output.cpu().numpy(); copied_out = time.perf_counter()
            normalize(output); ended = time.perf_counter()
            return ended - begun, {"tokenize_s": tokenized-begun, "h2d_s": copied_in-tokenized,
                                   "run_s": executed-copied_in, "d2h_s": copied_out-executed,
                                   "normalize_s": ended-copied_out}, list(inputs["input_ids"].shape)
        first_request_s, _, shape = once()
        for _ in range(warmup): once()
        samples, segments, shape = [], {}, None
        for _ in range(runs):
            value, part, shape = once(); samples.append(value)
            for name, seconds in part.items(): segments.setdefault(name, []).append(seconds)
        memory = benchmark_memory(device)
    return {"samples_s": samples, "model_load_s": model_load_s,
            "first_request_s": first_request_s, "memory": memory,
            "segments_p50_s": {name: percentile(values, 50) for name, values in segments.items()}, "actual_input_shape": shape}


def benchmark_ort(session, model_dir: Path, device_id: int, texts, warmup: int, runs: int,
                  profile_dir: Path) -> Dict[str, Any]:
    import onnxruntime as ort
    load_started = time.perf_counter()
    options = ort.SessionOptions()
    provider = ("CUDAExecutionProvider", {"device_id": device_id, "use_tf32": "0"})
    runtime = ort.InferenceSession(str(model_dir / "model.onnx"), sess_options=options, providers=[provider])
    model_load_s = time.perf_counter() - load_started
    if not runtime.get_providers() or runtime.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("ORT CUDA provider was not selected: {}".format(runtime.get_providers()))
    requests = benchmark_requests(texts)
    def once():
        begun = time.perf_counter(); inputs = token_inputs(session, next(requests)); tokenized = time.perf_counter()
        binding = runtime.io_binding()
        device_values = []
        for name, value in inputs.items():
            device_value = ort.OrtValue.ortvalue_from_numpy(value, "cuda", device_id)
            device_values.append(device_value)
            binding.bind_ortvalue_input(name, device_value)
        binding.bind_output("embeddings", "cuda", device_id)
        binding.synchronize_inputs(); copied_in = time.perf_counter()
        runtime.run_with_iobinding(binding); binding.synchronize_outputs(); executed = time.perf_counter()
        output = binding.copy_outputs_to_cpu()[0]; copied_out = time.perf_counter()
        normalize(output); ended = time.perf_counter()
        return ended-begun, {"tokenize_s": tokenized-begun, "h2d_s": copied_in-tokenized,
                             "run_s": executed-copied_in, "d2h_s": copied_out-executed,
                             "normalize_s": ended-copied_out}, list(inputs["input_ids"].shape)
    first_request_s, _, shape = once()
    for _ in range(warmup): once()
    samples, segments, shape = [], {}, None
    for _ in range(runs):
        value, part, shape = once(); samples.append(value)
        for name, seconds in part.items(): segments.setdefault(name, []).append(seconds)
    memory = benchmark_memory("nvidia")
    # Provider profiling uses a separate session so profiling overhead never
    # contaminates the formal latency samples above.
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_options = ort.SessionOptions(); profile_options.enable_profiling = True
    profile_options.profile_file_prefix = str((profile_dir / "ort-profile").resolve())
    profiled = ort.InferenceSession(str(model_dir / "model.onnx"), sess_options=profile_options, providers=[provider])
    inputs = token_inputs(session, next(requests)); binding = profiled.io_binding()
    device_values = []
    for name, value in inputs.items():
        device_value = ort.OrtValue.ortvalue_from_numpy(value, "cuda", device_id)
        device_values.append(device_value)
        binding.bind_ortvalue_input(name, device_value)
    binding.bind_output("embeddings", "cuda", device_id); profiled.run_with_iobinding(binding)
    profile_path = profiled.end_profiling()
    events = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    node_events = [event for event in events if event.get("cat") == "Node"
                   and event.get("args", {}).get("provider")]
    providers = sorted({event["args"]["provider"] for event in node_events})
    cuda_nodes = [event for event in node_events if event["args"]["provider"] == "CUDAExecutionProvider"]
    cpu_nodes = [event for event in node_events if event["args"]["provider"] == "CPUExecutionProvider"]
    metadata_ops = {"Shape", "Gather", "Slice", "Concat", "Unsqueeze", "Squeeze", "Cast",
                    "Add", "Sub", "Mul", "Div", "Equal", "Less", "Greater", "Where", "Range",
                    "ConstantOfShape", "Expand", "Reshape", "Identity"}
    integer_types = {"int64", "int32", "int16", "int8", "uint64", "uint32", "uint16", "uint8", "bool"}
    def metadata_node(event):
        detail = event["args"]
        values = detail.get("input_type_shape", []) + detail.get("output_type_shape", [])
        dtypes = {dtype for value in values for dtype in value}
        total_bytes = sum(int(detail.get(name, 0)) for name in
                          ("parameter_size", "activation_size", "output_size"))
        return detail.get("op_name") in metadata_ops and dtypes and dtypes <= integer_types and total_bytes <= 4096
    invalid_cpu = [event for event in cpu_nodes if not metadata_node(event)]
    other = [event for event in node_events if event["args"]["provider"] not in
             {"CUDAExecutionProvider", "CPUExecutionProvider"}]
    if not cuda_nodes or invalid_cpu or other:
        details = [(event.get("name"), event["args"].get("op_name"),
                    event["args"].get("provider"), event["args"].get("output_type_shape"))
                   for event in invalid_cpu + other]
        raise RuntimeError("ORT profiling found encoder data nodes outside CUDA: {}".format(details))
    cpu_metadata = [{"name": event.get("name"), "op_type": event["args"].get("op_name"),
                     "input_type_shape": event["args"].get("input_type_shape"),
                     "output_type_shape": event["args"].get("output_type_shape")}
                    for event in cpu_nodes]
    return {"samples_s": samples, "model_load_s": model_load_s,
            "first_request_s": first_request_s, "memory": memory,
            "segments_p50_s": {name: percentile(values, 50) for name, values in segments.items()},
            "actual_input_shape": shape, "ort_node_providers": providers,
            "ort_cuda_node_count": len(cuda_nodes),
            "ort_cpu_metadata_nodes": cpu_metadata, "ort_profile": profile_path,
            "ort_provider_options": runtime.get_provider_options()}


def benchmark_memory(device: str, session=None) -> Dict[str, Any]:
    answer: Dict[str, Any] = {"process_rss_peak_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024}
    if session is not None:
        answer["infinitensor"] = session.stub.memory_stats()
    try:
        import torch
        allocator = {"nvidia": getattr(torch, "cuda", None),
                     "cambricon": getattr(torch, "mlu", None),
                     "ascend": getattr(torch, "npu", None)}[device]
        if allocator is not None:
            if hasattr(allocator, "max_memory_allocated"):
                answer["aten_peak_allocated_bytes"] = int(allocator.max_memory_allocated())
            if hasattr(allocator, "max_memory_reserved"):
                answer["aten_peak_reserved_bytes"] = int(allocator.max_memory_reserved())
    except (AttributeError, RuntimeError):
        pass
    if device == "nvidia":
        try:
            rows = subprocess.check_output(
                ("nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"),
                text=True, stderr=subprocess.STDOUT, timeout=10)
            for row in rows.splitlines():
                pid, memory = (part.strip() for part in row.split(",", 1))
                if int(pid) == os.getpid():
                    answer["process_device_memory_mib"] = int(memory)
                    break
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    else:
        snapshot = resource_sample(device)
        if "process_device_memory_bytes" in snapshot:
            answer["process_device_memory_mib"] = snapshot["process_device_memory_bytes"] / 1024**2
        answer["external_device_load"] = snapshot.get("external_device_load")
    return answer


def check_benchmark(args, output: Path) -> Dict[str, Any]:
    # Each invocation is one independent round. The remote orchestrator starts
    # fresh processes in rotated engine order and combines their raw reports.
    load_started = time.perf_counter()
    if args.engine == "infinitensor":
        session = import_engine(args)
    else:
        from transformers import AutoTokenizer
        session = SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained(
            str(args.model_dir), local_files_only=True))
    frontend_load_s = time.perf_counter() - load_started
    report = {"engine": args.engine, "round": args.benchmark_round,
              "shared_environment": True, "process_entry_s": time.perf_counter() - PROCESS_STARTED,
              "frontend_load_s": frontend_load_s,
              "environment": device_env(args.device), "cases": []}
    selected_cases = [(8, 512)] if args.benchmark_mixed else ([tuple(args.benchmark_case)] if args.benchmark_case else CASES)
    for batch, length in selected_cases:
        if args.benchmark_mixed:
            rng = np.random.default_rng(315)
            texts = []
            for _ in range(1 + args.warmup + args.runs):
                longest = int(rng.integers(2, 513))
                lengths = [longest] + [int(rng.integers(2, longest + 1)) for _ in range(7)]
                texts.append(["测" * max(0, item - 2) for item in lengths])
        else:
            texts = ["测" * max(1, length - 2)] * batch
        if args.engine == "infinitensor":
            raw = benchmark_infinitensor(session, texts, batch, args.warmup, args.runs)
            actual_shape = list(session.stub.getShape("input_ids"))
        elif args.engine == "pytorch":
            raw = benchmark_torch(session, Path(args.model_dir), args.device, args.device_id, texts, args.warmup, args.runs)
            actual_shape = raw.pop("actual_input_shape")
        else:
            if args.device != "nvidia": raise ValueError("ORT CUDA benchmark is only valid on NVIDIA")
            raw = benchmark_ort(session, Path(args.model_dir), args.device_id, texts, args.warmup, args.runs, output)
            actual_shape = raw.pop("actual_input_shape")
        values = raw.pop("samples_s")
        measured_texts = texts[1 + args.warmup:] if args.benchmark_mixed else [texts] * args.runs
        measured_inputs = [token_inputs(session, request) for request in measured_texts]
        effective_counts = [int(item["attention_mask"].sum()) for item in measured_inputs]
        padded_counts = [int(item["attention_mask"].size) for item in measured_inputs]
        effective_tokens = float(np.mean(effective_counts))
        padded_tokens = float(np.mean(padded_counts))
        item = {"requested_shape": [batch, length], "samples": len(values),
                "p50_s": percentile(values, 50), "p95_s": percentile(values, 95),
                "texts_per_s": batch / percentile(values, 50),
                "effective_tokens_per_s": effective_tokens / percentile(values, 50),
                "padded_tokens_per_s": padded_tokens / percentile(values, 50),
                "tokens_per_s": padded_tokens / percentile(values, 50),
                "effective_tokens": effective_tokens, "padded_tokens": padded_tokens,
                "measured_input_shapes": [list(item["input_ids"].shape) for item in measured_inputs],
                "effective_token_counts": effective_counts, "padded_token_counts": padded_counts,
                "aggregate_texts_per_s": batch * len(values) / sum(values),
                "aggregate_effective_tokens_per_s": sum(effective_counts) / sum(values),
                "aggregate_padded_tokens_per_s": sum(padded_counts) / sum(values),
                "actual_input_shape": actual_shape, "raw_samples_s": values,
                "segments_p50_s": raw["segments_p50_s"],
                "first_request_s": raw["first_request_s"],
                "model_load_s": raw.get("model_load_s", frontend_load_s),
                "memory": raw.get("memory", benchmark_memory(args.device, session if args.engine == "infinitensor" else None))}
        if "ort_node_providers" in raw:
            item["ort_node_providers"] = raw["ort_node_providers"]
            item["ort_cuda_node_count"] = raw["ort_cuda_node_count"]
            item["ort_cpu_metadata_nodes"] = raw["ort_cpu_metadata_nodes"]
            item["ort_provider_options"] = raw["ort_provider_options"]
        if args.engine == "infinitensor":
            item["execution"] = session.execution_stats()
        report["cases"].append(item)
    report["environment_after"] = device_env(args.device)
    suffix = "-mixed" if args.benchmark_mixed else ("-{}x{}".format(*args.benchmark_case) if args.benchmark_case else "")
    write_json(output / "benchmark-{}-round-{}{}.json".format(args.engine, args.benchmark_round, suffix), report)
    return report


def check_install_stage(args, output: Path) -> Dict[str, Any]:
    """Verify the installed package from an offline, outside-tree process."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    session = import_engine(args)
    shapes = {name: (2, 17) for name in ("input_ids", "attention_mask", "token_type_ids")}
    capabilities = session.stub.check_capabilities(shapes)
    vectors = session.encode(["离线安装验证", "中文向量检索"], kind="passage")
    if vectors.shape != (2, 512) or not np.isfinite(vectors).all():
        raise AssertionError("installed package returned invalid vectors")
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5, rtol=0)
    evidence = source_evidence()
    native_paths = []
    maps = Path("/proc/self/maps")
    if maps.is_file():
        for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
            candidate = line.rsplit(None, 1)[-1]
            lowered = candidate.lower()
            basename = Path(candidate).name.lower()
            if candidate.startswith("/") and (any(name in basename for name in
                                                    ("libinfinitensor", "libinfiniops", "libinfinirt"))
                                                  or (basename.startswith("backend") and ".so" in basename)):
                native_paths.append(str(Path(candidate).resolve()))
        native_paths = sorted(set(native_paths))
    if args.install_prefix:
        prefix = args.install_prefix.resolve()
        outside = []
        for name, item in evidence.items():
            if not isinstance(item, dict) or "path" not in item or name == "accept_m1":
                continue
            try:
                Path(item["path"]).resolve().relative_to(prefix)
            except ValueError:
                outside.append(item["path"])
        for path in native_paths:
            try:
                Path(path).relative_to(prefix)
            except ValueError:
                outside.append(path)
        if outside:
            raise AssertionError("InfiniTensor modules are outside clean root {}: {}".format(prefix, outside))
    build_log = None
    if args.build_log:
        if not args.build_log.is_file():
            raise FileNotFoundError("clean build log is missing: {}".format(args.build_log))
        build_log = {"path": str(args.build_log.resolve()), "sha256": sha256(args.build_log)}
    report = {"status": "PASS", "offline": True, "module_evidence": evidence,
              "install_prefix": str(args.install_prefix.resolve()) if args.install_prefix else None,
              "build_log": build_log, "native_paths": native_paths,
              "capabilities": capabilities,
              "vector_norm_max_error": float(np.max(np.abs(norms - 1.0)))}
    write_json(output / "install.json", report)
    return report


def run_stage(args, output: Path, stage: str) -> Dict[str, Any]:
    return {"preflight": check_preflight, "accuracy": check_accuracy, "retrieval": check_retrieval,
            "stability": check_stability, "benchmark": check_benchmark,
            "install": check_install_stage}[stage](args, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "accuracy", "retrieval", "stability", "benchmark", "install", "all"), required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", choices=("nvidia", "cambricon", "ascend"), required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--download-dataset", action="store_true")
    parser.add_argument("--random-shapes", type=int, default=100)
    parser.add_argument("--stability-requests", type=int, default=10000)
    parser.add_argument("--stability-seconds", type=int, default=7200)
    parser.add_argument("--stability-repeat", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--benchmark-round", type=int, default=1)
    parser.add_argument("--benchmark-case", type=int, nargs=2, metavar=("B", "S"))
    parser.add_argument("--benchmark-mixed", action="store_true")
    parser.add_argument("--engine", choices=("infinitensor", "pytorch", "ort"), default="infinitensor")
    parser.add_argument("--execution-mode", choices=("normal", "auto", "graph"), default="auto")
    parser.add_argument("--graph-capture-after", type=int, default=4)
    parser.add_argument("--run-oom-probe", action="store_true")
    parser.add_argument("--install-prefix", type=Path)
    parser.add_argument("--build-log", type=Path)
    parser.add_argument("--oom-probe-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stability_repeat < 1:
        parser.error("--stability-repeat must be positive")
    if args.stage in {"stability", "all"} and args.execution_mode != "normal":
        required = args.graph_capture_after + 1 if args.execution_mode == "auto" else 2
        if args.stability_repeat < required:
            parser.error("--stability-repeat must allow at least one graph replay per shape")
    args.model_dir, args.dataset_dir = Path(args.model_dir), Path(args.dataset_dir or "m1-dataset")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.oom_probe_child:
        session = import_engine(args)
        shapes = {name: (8192, 512) for name in ("input_ids", "attention_mask", "token_type_ids")}
        session.stub.prepare(shapes)
        raise RuntimeError("OOM probe unexpectedly prepared an oversized BGE graph")
    envelope = {"stage": args.stage, "environment": device_env(args.device),
                "model": model_metadata(args.model_dir),
                "source_evidence": source_evidence(),
                "started_unix_s": time.time()}
    try:
        stages = ("preflight", "accuracy", "retrieval", "stability", "benchmark", "install") if args.stage == "all" else (args.stage,)
        envelope["results"] = {stage: run_stage(args, output, stage) for stage in stages}
        if args.run_oom_probe:
            command = [sys.executable, __file__, "--oom-probe-child", "--stage", "preflight", "--model-dir", str(args.model_dir),
                       "--device", args.device, "--device-id", str(args.device_id), "--output", str(output / "oom-child")]
            completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
            if completed.returncode == 0:
                raise AssertionError("isolated OOM probe unexpectedly succeeded")
            failure_text = (completed.stdout + "\n" + completed.stderr).lower()
            oom_markers = ("out of memory", "memory allocation", "allocate memory", "alloc failed",
                           "nomem", "error 2", "500001")
            matched = [marker for marker in oom_markers if marker in failure_text]
            if not matched:
                raise AssertionError("isolated OOM probe failed without an allocation error: {}".format(
                    completed.stderr[-2000:]))
            # The next newly created instance must remain usable after a child
            # allocation failure. The child shares no allocator state.
            recovered = import_engine(args).encode(["OOM 后恢复验证"])
            if recovered.shape != (1, 512) or not np.isfinite(recovered).all():
                raise AssertionError("new session failed after OOM probe")
            write_json(output / "oom-probe.json", {"status": "PASS", "child_returncode": completed.returncode,
                                                     "matched_allocation_markers": matched,
                                                     "child_stderr": completed.stderr[-4000:]})
        envelope["status"] = ("CALIBRATED" if any(
            value.get("status") == "CALIBRATED" for value in envelope["results"].values()) else "PASS")
    except BaseException as error:
        envelope.update(status="FAIL", error="{}: {}".format(type(error).__name__, error))
        raise
    finally:
        envelope["finished_unix_s"] = time.time()
        write_json(output / "accept-{}.json".format(args.stage), envelope)


if __name__ == "__main__":
    main()
