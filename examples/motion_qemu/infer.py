"""Run the complete model from raw inputs, without importing Torch."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import onnx
from pyinfinitensor import backend
from pyinfinitensor.onnx import OnnxStub
from export_model import make_model


def run(model_dir, execution, samples=None, chunk=None, check_oracle=False):
    data = np.load(model_dir / "dataset.npz", allow_pickle=False)
    saved = onnx.load(model_dir / f"motion_{execution}.onnx")
    weights = {w.name: onnx.numpy_helper.to_array(w) for w in saved.graph.initializer}
    size = chunk or saved.graph.input[0].type.tensor_type.shape.dim[0].dim_value
    if chunk is not None and not 1 <= chunk <= 1024:
        raise ValueError("chunk must be in [1,1024]")
    total = len(data["raw"]) if samples is None else samples
    if not 1 <= total <= len(data["raw"]):
        raise ValueError("Invalid sample count")
    state = np.zeros((3, 113))
    previous = np.zeros(21)
    reference_state, reference_previous = state.copy(), previous.copy()
    results = {n: [] for n in ["prediction", "features", "counts"]}
    graphs = {}
    start = time.perf_counter()
    for offset in range(0, total, size):
        n = min(size, total - offset)
        if n not in graphs:
            graphs[n] = OnnxStub(
                make_model(weights, n, execution),
                backend.cpu_runtime(),
                simplify_model=False,
            )
        graph = graphs[n]
        raw = data["raw"][offset : offset + n]
        for name, value in {
            "raw": raw,
            "previous": previous,
            "feedback": data["feedback"][offset : offset + n],
            "state_in": state,
        }.items():
            graph.inputs[name].copyin_numpy(np.ascontiguousarray(value))
        graph.run()
        out = {name: t.copyout_numpy() for name, t in graph.outputs.items()}
        state, previous = out["state_out"], out["last_input"]
        if check_oracle:
            from reference import reservoir

            expected, reference_state, reference_previous = reservoir(
                raw, reference_previous, reference_state, data["wi"], data["wr"]
            )
            np.testing.assert_array_equal(out["counts"], expected)
            np.testing.assert_allclose(state, reference_state, rtol=1e-12, atol=1e-8)
            np.testing.assert_array_equal(previous, reference_previous)
        for name in results:
            results[name].append(out[name])
    result = {name: np.concatenate(value) for name, value in results.items()}
    result.update(state_out=state, last_input=previous)
    np.testing.assert_array_equal(result["counts"], data["expected_counts"][:total])
    np.testing.assert_array_equal(result["features"], data["expected_features"][:total])
    np.testing.assert_allclose(
        result["prediction"], data["expected_prediction"][:total], rtol=1e-5, atol=1e-4
    )
    labels = data["labels"][:total]
    summary = {
        "execution": execution,
        "samples": total,
        "chunk": size,
        "seconds": time.perf_counter() - start,
        "counts_exact": True,
        "features_exact": True,
        "oracle_states_checked": check_oracle,
        "prediction_max_error": float(
            np.max(np.abs(result["prediction"] - data["expected_prediction"][:total]))
        ),
        "mean_nmse": float(
            np.mean(
                np.mean((result["prediction"] - labels) ** 2, axis=0)
                / data["labels"].var(0)
            )
        ),
        "torch_imported": "torch" in sys.modules,
    }
    if summary["torch_imported"]:
        raise AssertionError("Inference imported Torch")
    return result, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--execution", choices=["reference", "qemu"], required=True)
    p.add_argument("--samples", type=int)
    p.add_argument("--chunk", type=int)
    p.add_argument("--check-oracle", action="store_true")
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    result, summary = run(a.model_dir, a.execution, a.samples, a.chunk, a.check_oracle)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output_dir / "outputs.npz", **result)
    (a.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
