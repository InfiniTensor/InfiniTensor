#!/usr/bin/env python3
"""Verify the dynamic BGE path and record an initial shared-device baseline."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def normalized(value):
    value = np.asarray(value, dtype=np.float32)
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    assert np.all(np.isfinite(norm)) and np.all(norm > 0)
    result = value / norm
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=1e-5, rtol=0)
    return result


def compare(actual, expected):
    actual, expected = normalized(actual), normalized(expected)
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3)
    cosine = np.sum(actual * expected, axis=1)
    assert np.min(cosine) >= 0.99999, float(np.min(cosine))
    return {
        "max_abs": float(np.max(np.abs(actual - expected))),
        "min_cosine": float(np.min(cosine)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--random-shapes", type=int, default=100)
    parser.add_argument("--benchmark-iterations", type=int, default=50)
    parser.add_argument("--trace-only", action="store_true")
    args = parser.parse_args()
    assert args.random_shapes >= 0 and args.benchmark_iterations > 0
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "benchmark_scope": "shared device; one process; initial baseline, not full M1 performance acceptance",
    }

    def save():
        (args.output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    try:
        import torch
        import onnxruntime as ort
        from transformers import AutoModel
        from pyinfinitensor.embedding import (
            EmbeddingSession,
            BGE_QUERY_PREFIX,
            fp32_inference_policy,
        )

        manifest = json.loads((args.model_dir / "manifest.json").read_text())
        for name, expected in manifest["sha256"].items():
            path = args.model_dir / name
            assert path.resolve().is_relative_to(args.model_dir.resolve())
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
        report["model_revision"] = manifest["revision"]
        report["model_sha256"] = manifest["sha256"]["model.onnx"]
        torch.manual_seed(315)
        rng = np.random.default_rng(315)
        torch.cuda.set_device(args.device_id)
        started = time.perf_counter()
        session = EmbeddingSession(
            args.model_dir, device="nvidia", device_id=args.device_id
        )
        report["load_s"] = time.perf_counter() - started
        report["environment"] = {
            "torch": torch.__version__,
            "ort": ort.__version__,
            "device": torch.cuda.get_device_name(args.device_id),
            "cuda": torch.version.cuda,
        }
        stub = session.stub
        weight_state = (
            stub.weight_upload_count,
            stub.weight_upload_bytes,
            stub.weight_addresses,
        )

        def inputs(batch, sequence):
            result = {
                "input_ids": rng.integers(1, 1000, (batch, sequence), dtype=np.int64),
                "attention_mask": np.ones((batch, sequence), dtype=np.int64),
                "token_type_ids": np.zeros((batch, sequence), dtype=np.int64),
            }
            lengths = rng.integers(1, sequence + 1, batch)
            for row, length in enumerate(lengths):
                result["attention_mask"][row, length:] = 0
                result["input_ids"][row, length:] = 0
            return result

        def run_infini(values):
            with fp32_inference_policy():
                return next(iter(stub.run_numpy(values).values()))

        if args.trace_only:
            assert run_infini(inputs(1, 17)).shape == (1, 512)
            report["scope"] = "single forward for InfiniOps device-placement trace only"
            report["status"] = "passed"
            save()
            return

        from onnx import helper, numpy_helper, TensorProto
        from pyinfinitensor import backend
        from pyinfinitensor.onnx import OnnxStub

        # Check the changed Slice bridge numerically on the real device too.
        graph = helper.make_graph(
            [
                helper.make_node("Identity", ["x"], ["middle"]),
                helper.make_node("Slice", ["middle", "start", "end"], ["y"]),
            ],
            "dynamic_slice",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["S"])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None])],
            [
                numpy_helper.from_array(np.array([0], np.int64), "start"),
                numpy_helper.from_array(np.array([2], np.int64), "end"),
            ],
        )
        small = OnnxStub(
            helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
            backend.runtime("nvidia", args.device_id),
        )
        for length in (1, 3, 2, 1):
            value = np.arange(length, dtype=np.float32)
            np.testing.assert_array_equal(small.run_numpy({"x": value})["y"], value[:2])
        report["slice_device_regression"] = "passed"
        del small

        reference = (
            AutoModel.from_pretrained(
                str(args.model_dir), local_files_only=True, attn_implementation="eager"
            )
            .eval()
            .cuda(args.device_id)
        )

        def run_torch(values):
            with fp32_inference_policy(), torch.inference_mode():
                tensors = {
                    name: torch.from_numpy(value).cuda(args.device_id)
                    for name, value in values.items()
                }
                return reference(**tensors).last_hidden_state[:, 0, :].cpu().numpy()

        providers = [
            ("CUDAExecutionProvider", {"device_id": args.device_id, "use_tf32": "0"}),
            "CPUExecutionProvider",
        ]
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str(args.output / "ort-provider")
        ort_session = ort.InferenceSession(
            str(session.model_path), sess_options=options, providers=providers
        )
        assert "CUDAExecutionProvider" in ort_session.get_providers()

        def ort_run(engine, values):
            binding = engine.io_binding()
            for name, value in values.items():
                binding.bind_cpu_input(name, value)
            for output in engine.get_outputs():
                binding.bind_output(output.name, "cuda", args.device_id)
            engine.run_with_iobinding(binding)
            return binding.copy_outputs_to_cpu()[0]

        initial = inputs(1, 17)
        # Establish export consistency before judging the new engine.
        np.testing.assert_allclose(
            ort_run(ort_session, initial), run_torch(initial), atol=1e-4, rtol=1e-3
        )
        profile = Path(ort_session.end_profiling())
        nodes = [
            e
            for e in json.loads(profile.read_text())
            if e.get("cat") == "Node" and e.get("args", {}).get("provider")
        ]
        assert nodes, "ORT profiling returned no node placement evidence"
        for event in nodes:
            data = event["args"]
            if (
                data["provider"] != "CUDAExecutionProvider"
                and data.get("op_name") != "Shape"
            ):
                types = json.dumps(
                    [data.get("input_type_shape"), data.get("output_type_shape")]
                )
                assert "float" not in types, event
        report["ort_nodes"] = [
            {
                "name": e["name"],
                "op": e["args"].get("op_name"),
                "provider": e["args"]["provider"],
            }
            for e in nodes
        ]
        del ort_session
        ort_session = ort.InferenceSession(str(session.model_path), providers=providers)

        shapes = [(1, 17), (7, 83), (3, 511), (16, 128), (1, 17)]
        shapes += [
            (b, s) for b in (1, 3, 8, 17, 32) for s in (2, 17, 64, 127, 255, 511, 512)
        ]
        shapes += [(1, 1), (32, 1), (32, 512), (1, 17), (1, 17)]
        shapes += [
            (int(rng.integers(1, 33)), int(rng.integers(1, 513)))
            for _ in range(args.random_shapes)
        ]
        report["shapes"] = []
        previous = None
        with (args.output / "progress.jsonl").open("w", encoding="utf-8") as progress:
            for batch, sequence in shapes:
                values = inputs(batch, sequence)
                actual = run_infini(values)
                if previous is not None:
                    np.testing.assert_array_equal(*previous)
                assert actual.shape == (batch, 512) and actual.dtype == np.float32
                result = {
                    "shape": [batch, sequence],
                    "torch": compare(actual, run_torch(values)),
                    "ort": compare(actual, ort_run(ort_session, values)),
                }
                assert weight_state == (
                    stub.weight_upload_count,
                    stub.weight_upload_bytes,
                    stub.weight_addresses,
                )
                previous = (actual, actual.copy())
                report["shapes"].append(result)
                progress.write(json.dumps(result) + "\n")
                progress.flush()
                print("verified", batch, sequence, flush=True)

        bad = inputs(1, 17)
        for values in (
            {"wrong": bad["input_ids"]},
            dict(bad, input_ids=bad["input_ids"].astype(np.int32)),
            dict(bad, attention_mask=bad["attention_mask"][:, :2]),
        ):
            try:
                stub.run_numpy(values)
            except (ValueError, TypeError):
                pass
            else:
                raise AssertionError("invalid inputs were accepted")
        compare(run_infini(bad), run_torch(bad))

        texts = ["中文知识库检索", "混合 English 和标点！", "", "长文本" * 600] + [
            "条目{}".format(i) for i in range(33)
        ]
        for kind in ("query", "passage"):
            actual = session.encode(texts, kind=kind)
            np.testing.assert_allclose(
                np.linalg.norm(actual, axis=1), 1, atol=1e-5, rtol=0
            )
            prepared = [BGE_QUERY_PREFIX + t if kind == "query" else t for t in texts]
            expected = []
            for start in range(0, len(texts), 32):
                values = dict(
                    session.tokenizer(
                        prepared[start : start + 32],
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="np",
                        return_token_type_ids=True,
                    )
                )
                expected.append(normalized(run_torch(values)))
            compare(actual, np.concatenate(expected))
        assert session.encode([]).shape == (0, 512)
        compare(session.encode(texts[:1]), session.encode([texts[0], texts[3]])[:1])
        assert weight_state == (
            stub.weight_upload_count,
            stub.weight_upload_bytes,
            stub.weight_addresses,
        )
        report["text_cases"] = (
            "passed: batching/order, prefix, truncation, empty list/string, independent vs padded"
        )
        report["weight_upload_count"] = stub.weight_upload_count
        report["weight_upload_bytes"] = stub.weight_upload_bytes
        report["benchmark"] = []
        for batch, sequence in [(1, 32), (1, 128), (1, 512), (8, 128), (32, 128)]:
            values = inputs(batch, sequence)
            row = {"shape": [batch, sequence], "engines": {}}
            for name, engine in [
                ("InfiniTensor", run_infini),
                ("PyTorch", run_torch),
                ("ORT CUDA", lambda x: ort_run(ort_session, x)),
            ]:
                for _ in range(20):
                    engine(values)
                samples = []
                for _ in range(args.benchmark_iterations):
                    torch.cuda.synchronize(args.device_id)
                    start = time.perf_counter()
                    normalized(engine(values))
                    torch.cuda.synchronize(args.device_id)
                    samples.append(time.perf_counter() - start)
                row["engines"][name] = {
                    "samples_s": samples,
                    "p50_ms": float(np.median(samples) * 1000),
                    "p95_ms": float(np.percentile(samples, 95) * 1000),
                    "texts_per_s": float(batch / np.median(samples)),
                }
            report["benchmark"].append(row)
            save()
            print("benchmarked", batch, sequence, flush=True)
        report["status"] = "passed"
        save()
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        save()
        raise


if __name__ == "__main__":
    main()
