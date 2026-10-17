#!/usr/bin/env python3
"""Aggregate the six M1 endpoints and emit immutable provenance evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, Iterable, List


ENDPOINTS = ("l20", "a100", "mlu1", "mlu2", "npu1", "npu2")
REQUIRED = ("preflight", "accuracy", "retrieval", "stability", "oom", "benchmark", "install")
ENGINE_PARTS = ("CMakeLists.txt", "Makefile", "src", "include", "pyinfinitensor/src", "dependencies")
DELIVERY_PARTS = ENGINE_PARTS + ("env.sh", "pyinfinitensor/pyproject.toml", "pyinfinitensor/tests", "test", "scripts", "examples/python", "docs")
# Source provenance must remain stable across platform builds.  Native modules
# copied into the Python package are installation artifacts and are recorded by
# the install stage, not as source files.
IGNORED_SUFFIXES = (".pyc", ".pyo", ".so", ".a", ".dll", ".dylib", ".pyd")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {"value": value}
    except (OSError, ValueError) as error:
        return {"status": "FAIL", "parse_error": repr(error)}


def status(path: Path) -> str:
    value = read_json(path)
    return str(value.get("status", "MISSING"))


def source_manifest(source: Path, parts: Iterable[str]) -> Dict[str, Any]:
    files: Dict[str, str] = {}
    for part in parts:
        base = source / part
        if not base.exists():
            continue
        candidates: Iterable[Path] = (base.rglob("*") if base.is_dir() else (base,))
        for path in sorted(candidates):
            if (not path.is_file() or path.suffix in IGNORED_SUFFIXES
                    or "__pycache__" in path.parts):
                continue
            files[path.relative_to(source).as_posix()] = sha256(path)
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"tree_sha256": hashlib.sha256(canonical).hexdigest(), "files": files}


def benchmark_summary(root: Path) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
    provider_sets = set()
    for path in sorted(root.glob("benchmark-*-round-*.json")):
        if path.name == "benchmark-matrix.json":
            continue
        report = read_json(path)
        engine = str(report.get("engine", "unknown"))
        for item in report.get("cases", []):
            requested = item.get("requested_shape", [])
            case = "mixed" if path.stem.endswith("-mixed") else "x".join(map(str, requested))
            target = grouped.setdefault(engine, {}).setdefault(case, [])
            if "p50_s" in item:
                target.append({key: float(item[key]) for key in
                               ("p50_s", "p95_s", "texts_per_s", "effective_tokens_per_s",
                                "padded_tokens_per_s") if key in item})
            for provider in item.get("ort_node_providers", []):
                provider_sets.add(provider)
    values = {engine: {case: {"rounds": len(samples),
                              "median_p50_ms": statistics.median(item["p50_s"] for item in samples) * 1000,
                              "median_p95_ms": statistics.median(item["p95_s"] for item in samples) * 1000,
                              "median_texts_per_s": statistics.median(item["texts_per_s"] for item in samples),
                              "median_effective_tokens_per_s": statistics.median(item["effective_tokens_per_s"] for item in samples),
                              "median_padded_tokens_per_s": statistics.median(item["padded_tokens_per_s"] for item in samples)}
                       for case, samples in cases.items() if samples}
              for engine, cases in grouped.items()}
    return {"cases": values, "ort_node_providers": sorted(provider_sets)}


def endpoint_result(root: Path) -> Dict[str, Any]:
    preflight = read_json(root / "accept-preflight.json")
    accuracy = read_json(root / "accuracy.json")
    retrieval = read_json(root / "retrieval.json")
    stability = read_json(root / "stability.json")
    stage_status = {
        "preflight": status(root / "accept-preflight.json"),
        "accuracy": status(root / "accept-accuracy.json"),
        "retrieval": status(root / "accept-retrieval.json"),
        "stability": status(root / "accept-stability.json"),
        "oom": status(root / "oom-probe.json"),
        "benchmark": status(root / "benchmark-matrix.json"),
        "install": status(root / "install.json"),
    }
    if (stability.get("status") != "PASS" or stability.get("method_version") != 2
            or stability.get("requests", 0) < 10000 or stability.get("elapsed_s", 0) < 7200
            or stability.get("reference_checks", 0) < stability.get("requests", 0) // 100
            or not str(stability.get("reference_engine", "")).startswith("pytorch-")):
        stage_status["stability"] = "FAIL"
    regressions = {}
    for name in ("ctest.txt", "infinirt-ctest.txt", "test_embedding.log", "test_onnxstub.log",
                 "test_accept_m1.log", "test_copy_graph_capture.log", "test_workspace_graph_capture.log"):
        text = (root / name).read_text(errors="replace") if (root / name).is_file() else ""
        regressions[name] = bool(text) and ("100% tests passed" in text if name.endswith(".txt") else "\nOK" in text)
    accelerator_ops = read_json(root / "accelerator-ops.json")
    regressions["accelerator-ops.json"] = accelerator_ops.get("status") == "PASS"
    if not all(regressions.values()):
        stage_status["install"] = "FAIL"
    return {
        "status": "PASS" if all(stage_status[name] == "PASS" for name in REQUIRED) else "FAIL",
        "stages": stage_status,
        "accuracy": {key: accuracy.get(key) for key in
                     ("requests", "min_cosine", "max_abs_error", "weight_upload_count",
                      "weight_upload_bytes", "weight_addresses_stable")},
        "retrieval": {key: retrieval.get(key) for key in
                      ("mean_top10_overlap", "min_vector_cosine", "max_vector_abs_error",
                       "selection_sha256")},
        "stability": {key: stability.get(key) for key in
                      ("requests", "elapsed_s", "weight_upload_count", "weight_addresses_stable")},
        "resource_evidence": stability.get("resource_evidence"),
        "environment": preflight.get("environment"),
        "source_evidence": preflight.get("source_evidence"),
        "regressions": regressions,
        "benchmark": benchmark_summary(root),
    }


def markdown_report(result: Dict[str, Any]) -> str:
    lines = [
        "# InfiniTensor M1 最终验收报告",
        "",
        "本报告由六端点原始 JSON 自动聚合。性能数据来自共享负载环境，只用于记录当前实现的基线，",
        "不用于宣称相对 PyTorch、ORT 或其他引擎的独占设备性能优势。",
        "",
        "## 版本与证据",
        "",
        "| 项目 | 内容摘要 |",
        "|---|---|",
        "| 总体状态 | **{}** |".format(result["status"]),
        "| 推理执行源码树 SHA256 | `{}` |".format(result["source"]["engine"]["tree_sha256"]),
        "| 完整交付源码树 SHA256 | `{}` |".format(result["source"]["delivery"]["tree_sha256"]),
        "| InfiniTensor 基线 | `{}` |".format(result["lock"].get("InfiniTensor_base")),
        "| InfiniRT | `{}` |".format(result["lock"].get("InfiniRT")),
        "| InfiniOps 基线 | `{}` |".format(result["lock"].get("InfiniOps_base")),
        "| InfiniOps 补丁 SHA256 | `{}` |".format(result["infiniops_patch_sha256"]),
        "| ONNX SHA256 | `{}` |".format(result.get("model_onnx_sha256")),
        "| 数据集 revision | `{}` |".format(result["dataset"].get("revision")),
        "",
        "## 六端点验收矩阵",
        "",
        "| 端点 | 预检 | 精度 | 检索 | 长稳 | OOM | 性能 | 安装 | 结论 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in ENDPOINTS:
        item = result["endpoints"][name]
        stage = item["stages"]
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | **{}** |".format(
            name, stage["preflight"], stage["accuracy"], stage["retrieval"],
            stage["stability"], stage["oom"], stage["benchmark"], stage["install"], item["status"]))
    lines += ["", "## 核心数值", "",
              "| 端点 | 动态用例 | 最小余弦 | 最大误差 | Top-10 重合率 | 长稳请求 | 长稳时长/秒 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ENDPOINTS:
        item = result["endpoints"][name]
        acc, ret, stable = item["accuracy"], item["retrieval"], item["stability"]
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            name, acc.get("requests"), acc.get("min_cosine"), acc.get("max_abs_error"),
            ret.get("mean_top10_overlap"), stable.get("requests"), stable.get("elapsed_s")))
    lines += ["", "## 共享环境性能基线", "",
              "| 端点 | 引擎 | 输入 | 轮数 | p50/ms | p95/ms | texts/s | 有效 tokens/s |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for name in ENDPOINTS:
        for engine, cases in result["endpoints"][name]["benchmark"]["cases"].items():
            for case, metrics in cases.items():
                lines.append("| {} | {} | {} | {} | {:.4f} | {:.4f} | {:.2f} | {:.2f} |".format(
                    name, engine, case, metrics["rounds"], metrics["median_p50_ms"],
                    metrics["median_p95_ms"], metrics["median_texts_per_s"],
                    metrics["median_effective_tokens_per_s"]))
    lines += ["", "ORT 的 CPU provider 只允许执行报告中列出的整数 shape 元数据节点；任何浮点 encoder 节点不在 CUDA 都会失败。",
              "MLU/NPU 不运行 ORT CUDA 对照。所有数字来自用户指定的共享负载环境。", ""]
    lines += [
        "",
        "## 完成判定",
        "",
        "只有六端点七个必需阶段全部为 PASS 时，总体状态才是 PASS。缺少报告、JSON 无法解析或任一阶段失败，",
        "聚合器都会给出 FAIL 并以非零状态退出。各端点原始样本、provider 分配、资源采样、构建日志和",
        "安装记录保留在结果目录中，表格仅是索引。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    lock_path = args.source / "dependencies" / "m1-lock.json"
    lock = read_json(lock_path)
    patch_path = args.source / "dependencies" / str(lock.get("InfiniOps_patch", "InfiniOps-m1.patch"))
    patch_hash = sha256(patch_path) if patch_path.is_file() else None
    if patch_hash != lock.get("InfiniOps_patch_sha256"):
        raise RuntimeError("InfiniOps patch hash does not match m1-lock.json")
    model_hash = sha256(args.model_dir / "model.onnx") if args.model_dir and (args.model_dir / "model.onnx").is_file() else None
    dataset: Dict[str, Any] = {"revision": "8731a845f1bf500a4f111cf1070785c793d10e64", "files": {}}
    if args.dataset_dir:
        for path in sorted(args.dataset_dir.glob("*.parquet")):
            dataset["files"][path.name] = sha256(path)
    endpoints = {name: endpoint_result(args.results_root / name) for name in ENDPOINTS}
    engine_manifest = source_manifest(args.source, ENGINE_PARTS)
    delivery_manifest = source_manifest(args.source, DELIVERY_PARTS)
    accept_hash = sha256(args.source / "scripts/accept_m1.py")
    for name, item in endpoints.items():
        actual = item.get("source_evidence", {}).get("engine_tree_sha256")
        item["source_matches_release"] = actual == engine_manifest["tree_sha256"]
        if not item["source_matches_release"]:
            item["status"] = "FAIL"
        mismatches = []
        for stage in ("preflight", "accuracy", "retrieval", "stability", "install"):
            evidence = read_json(args.results_root / name / ("accept-" + stage + ".json")).get("source_evidence", {})
            if (evidence.get("engine_tree_sha256") != engine_manifest["tree_sha256"]
                    or evidence.get("delivery_tree_sha256") != delivery_manifest["tree_sha256"]
                    or evidence.get("accept_m1", {}).get("sha256") != accept_hash):
                mismatches.append(stage)
        item["source_mismatch_stages"] = mismatches
        if mismatches:
            item["status"] = "FAIL"
    result = {
        "status": "PASS" if all(item["status"] == "PASS" for item in endpoints.values()) else "FAIL",
        "lock": lock,
        "infiniops_patch_sha256": patch_hash,
        "source": {"engine": engine_manifest,
                   "delivery": delivery_manifest},
        "model_onnx_sha256": model_hash,
        "dataset": dataset,
        "endpoints": endpoints,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    args.output_md.write_text(markdown_report(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output_json)}, ensure_ascii=False))
    if result["status"] != "PASS" and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
