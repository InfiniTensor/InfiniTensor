"""Compare complete CPU/QEMU inference against independent supplied-model oracles."""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import onnx
from pyinfinitensor import backend
from pyinfinitensor.onnx import OnnxStub
from export_model import make_model
from infer import run
from reference import reservoir


def audit(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def boundary_checks(weights):
    raw = np.zeros((8, 21), np.float64)
    raw[1, 0] = 1e-5
    raw[2, 0] = -1e-5
    raw[3, :] = 0.001
    raw[4, :] = -0.001
    raw[5, :] = 1e-10
    feedback = np.zeros((8, 7))
    state = np.zeros((3, 113))
    state[0] = np.linspace(-17, 17, 113)
    state[1] = np.linspace(-32, 32, 113)
    state[2, ::3] = 0.5
    counts, expected_state, _ = reservoir(
        raw, np.zeros(21), state, weights["wi"], weights["wr"]
    )
    outputs = {}
    for mode in ["reference", "qemu"]:
        for chunk in [8, 3]:
            carried = state.copy()
            previous = np.zeros(21)
            all_counts = []
            all_features = []
            for start in range(0, 8, chunk):
                n = min(chunk, 8 - start)
                graph = OnnxStub(
                    make_model(weights, n, mode),
                    backend.cpu_runtime(),
                    simplify_model=False,
                )
                for name, value in {
                    "raw": raw[start : start + n],
                    "previous": previous,
                    "feedback": feedback[start : start + n],
                    "state_in": carried,
                }.items():
                    graph.inputs[name].copyin_numpy(np.ascontiguousarray(value))
                graph.run()
                carried = graph.outputs["state_out"].copyout_numpy()
                previous = graph.outputs["last_input"].copyout_numpy()
                all_counts.append(graph.outputs["counts"].copyout_numpy())
                all_features.append(graph.outputs["features"].copyout_numpy())
            np.testing.assert_array_equal(np.concatenate(all_counts), counts)
            np.testing.assert_allclose(carried, expected_state, rtol=1e-12, atol=1e-8)
            outputs[mode, chunk] = np.concatenate(all_features)
        np.testing.assert_array_equal(outputs[mode, 8], outputs[mode, 3])
        bad = weights.copy()
        bad["count_scale"] = np.zeros(113)
        graph = OnnxStub(
            make_model(bad, 1, mode), backend.cpu_runtime(), simplify_model=False
        )
        for name, value in {
            "raw": raw[:1],
            "previous": np.zeros(21),
            "feedback": feedback[:1],
            "state_in": state,
        }.items():
            graph.inputs[name].copyin_numpy(np.ascontiguousarray(value))
        try:
            graph.run()
        except RuntimeError as error:
            if "scale must be positive" not in str(error):
                raise
        else:
            raise AssertionError("Invalid scale was accepted")
        graph = OnnxStub(
            make_model(weights, 1, mode), backend.cpu_runtime(), simplify_model=False
        )
        bad_state = state.copy()
        bad_state[0, 0] = np.nan
        for name, value in {
            "raw": raw[:1],
            "previous": np.zeros(21),
            "feedback": feedback[:1],
            "state_in": bad_state,
        }.items():
            graph.inputs[name].copyin_numpy(np.ascontiguousarray(value))
        try:
            graph.run()
        except RuntimeError as error:
            if "non-finite" not in str(error):
                raise
        else:
            raise AssertionError("Non-finite state was accepted")
    bad = weights.copy()
    bad["wi"] = np.zeros((41, 113))
    try:
        OnnxStub(
            make_model(bad, 1, "reference"), backend.cpu_runtime(), simplify_model=False
        )
    except RuntimeError as error:
        if "invalid shape or dtype" not in str(error):
            raise
    else:
        raise AssertionError("Invalid weight shape was accepted")
    np.testing.assert_array_equal(outputs["reference", 8], outputs["qemu", 8])
    return {
        "negative_and_zero_inputs": True,
        "fractional_negative_currents": True,
        "floating_refractory": True,
        "split_state_carry": True,
        "invalid_scale_rejected_both_modes": True,
        "nonfinite_state_rejected_both_modes": True,
        "invalid_weight_shape_rejected": True,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--samples", type=int)
    p.add_argument("--chunk", type=int, default=64)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    path = a.output_dir / "qemu_audit.jsonl"
    os.environ["INFINI_RVNE_AUDIT"] = str(path)
    initial = len(audit(path))
    cpu, cpu_summary = run(a.model_dir, "reference", a.samples, a.chunk, True)
    assert len(audit(path)) == initial, "CPU path launched QEMU"
    np.savez_compressed(a.output_dir / "reference_outputs.npz", **cpu)
    print("CPU", json.dumps(cpu_summary), flush=True)
    qemu, qemu_summary = run(a.model_dir, "qemu", a.samples, a.chunk, False)
    entries = audit(path)[initial:]
    expected = (cpu_summary["samples"] + a.chunk - 1) // a.chunk
    assert len(entries) == expected
    for entry in entries:
        assert entry["status"] == "ok"
        assert [r["command"] for r in entry["requests"]] == [
            "MotionFeatures",
            "MotionReadout",
        ]
        assert all(r["status"] == "ok" for r in entry["requests"])
    for name in ["counts", "features", "last_input"]:
        np.testing.assert_array_equal(cpu[name], qemu[name])
    np.testing.assert_allclose(
        cpu["state_out"], qemu["state_out"], rtol=1e-12, atol=1e-8
    )
    np.testing.assert_allclose(
        cpu["prediction"], qemu["prediction"], rtol=1e-5, atol=1e-4
    )
    np.savez_compressed(a.output_dir / "qemu_outputs.npz", **qemu)
    saved = onnx.load(a.model_dir / "motion_reference.onnx")
    weights = {w.name: onnx.numpy_helper.to_array(w) for w in saved.graph.initializer}
    summary = {
        "cpu": cpu_summary,
        "qemu": qemu_summary,
        "qemu_launches": len(entries),
        "commands_per_graph_run": 2,
        "counts_and_features_exact_between_paths": True,
        "prediction_max_error_between_paths": float(
            np.max(np.abs(cpu["prediction"] - qemu["prediction"]))
        ),
        "boundary_checks": boundary_checks(weights),
    }
    (a.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
