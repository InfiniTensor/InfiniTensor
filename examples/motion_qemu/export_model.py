"""Export the supplied Baxter model; Torch is needed only for checkpoint import."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh

DOMAIN = "com.infinitensor.rvne"


def make_model(weights, samples, execution):
    inputs = [
        h.make_tensor_value_info(n, T.DOUBLE, s)
        for n, s in [
            ("raw", [samples, 21]),
            ("previous", [21]),
            ("feedback", [samples, 7]),
            ("state_in", [3, 113]),
        ]
    ]
    feature_inputs = [
        "raw",
        "previous",
        "feedback",
        "state_in",
        "wi",
        "wr",
        "count_mean",
        "count_scale",
    ]
    nodes = [
        h.make_node(
            "MotionFeatures",
            feature_inputs,
            ["features", "state_out", "last_input", "counts"],
            domain=DOMAIN,
            execution=execution,
        ),
        h.make_node(
            "MotionReadout",
            ["features", "w1", "b1", "w2", "b2", "y_mean", "y_scale"],
            ["prediction"],
            domain=DOMAIN,
            execution=execution,
        ),
    ]
    outputs = [
        h.make_tensor_value_info(n, t, s)
        for n, t, s in [
            ("prediction", T.DOUBLE, [samples, 7]),
            ("features", T.FLOAT, [samples, 120]),
            ("state_out", T.DOUBLE, [3, 113]),
            ("last_input", T.DOUBLE, [21]),
            ("counts", T.INT32, [samples, 113]),
        ]
    ]
    model = h.make_model(
        h.make_graph(
            nodes,
            "BaxterMotion",
            inputs,
            outputs,
            [nh.from_array(np.ascontiguousarray(v), n) for n, v in weights.items()],
        ),
        opset_imports=[h.make_opsetid("", 13), h.make_opsetid(DOMAIN, 1)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def main():
    import torch
    from scipy.io import loadmat

    p = argparse.ArgumentParser()
    p.add_argument("--materials", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--chunk", type=int, default=64)
    a = p.parse_args()
    if not 1 <= a.chunk <= 1024:
        p.error("chunk must be in [1,1024]")
    with zipfile.ZipFile(a.materials / "SpikingQ/Datasets/motion_control.zip") as z:

        def load(n):
            return np.load(
                io.BytesIO(z.read("motion_control/" + n)), allow_pickle=False
            )

        mat = loadmat(io.BytesIO(z.read("motion_control/BaxterRand.mat")))
        train_counts = load("simulation_archive/re_count113.npy")
        counts = load("simulation_archive/re_count_test113.npy").astype(np.int32)
        expected_features = load("baxter_rand_testing.npy").astype(np.float32)
        weights = {
            "wi": load("simulation_archive/weight_IN_matrix_113.npy"),
            "wr": load("simulation_archive/weight_matrix_liquid_113.npy"),
            "count_mean": train_counts.mean(0),
            "count_scale": train_counts.std(0),
            "y_mean": mat["Y_train"].mean(0),
            "y_scale": mat["Y_train"].std(0),
        }
    weights["count_scale"][weights["count_scale"] == 0] = 1
    with zipfile.ZipFile(a.materials / "SpikingQ/SpikingQ1020.zip") as z:
        names = [
            n for n in z.namelist() if n.endswith("checkpoints/motion_control/last.pt")
        ]
        if len(names) != 1:
            raise ValueError("Expected one motion checkpoint")
        checkpoint_bytes = z.read(names[0])
    state = torch.load(
        io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True
    )["model"]
    weights.update(
        w1=state["fc1.weight"].T.contiguous().numpy(),
        b1=state["fc1.bias"].numpy(),
        w2=state["fc2.weight"].T.contiguous().numpy(),
        b2=state["fc2.bias"].numpy(),
    )
    with torch.inference_mode():
        x = torch.from_numpy(expected_features)
        y = torch.nn.functional.linear(
            torch.tanh(
                torch.nn.functional.linear(x, state["fc1.weight"], state["fc1.bias"])
            ),
            state["fc2.weight"],
            state["fc2.bias"],
        ).numpy()
    prediction = y * weights["y_scale"] + weights["y_mean"]
    feedback = np.concatenate(
        [
            np.zeros((1, 7)),
            ((mat["Y_test"] - weights["y_mean"]) / weights["y_scale"])[:-1],
        ]
    )
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in ["reference", "qemu"]:
        onnx.save(
            make_model(weights, a.chunk, mode), a.output_dir / f"motion_{mode}.onnx"
        )
    np.savez_compressed(
        a.output_dir / "dataset.npz",
        raw=mat["X_test"],
        feedback=feedback,
        labels=mat["Y_test"],
        expected_counts=counts,
        expected_features=expected_features,
        expected_prediction=prediction,
        **weights,
    )
    manifest = {
        "contract_version": 1,
        "samples": len(feedback),
        "chunk": a.chunk,
        "steps": 50,
        "neurons": 113,
        "state_rows": ["current", "voltage", "refractory"],
        "feedback": "previous standardized ground truth; initial standardized zero",
        "sample_boundary": "carry I/V/refractory and previous raw input; reset feedback spikes/counts",
        "dtypes": "double reservoir, float32 Linear/Tanh/Linear, double output denormalization",
        "execution": "two nodes, one QEMU session per graph.run; ordinary RISC-V floating point",
        "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "reference_nmse": float(
            np.mean(
                np.mean((prediction - mat["Y_test"]) ** 2, axis=0)
                / mat["Y_test"].var(0)
            )
        ),
    }
    (a.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
