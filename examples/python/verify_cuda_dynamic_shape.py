"""Direction 3 acceptance: dynamic shape on CUDA, checked against ONNX Runtime.

The same model instance is handed a series of different shapes without being
reloaded, and every output is compared with what ONNX Runtime makes of the same
model and the same input. Both the plain graph and the graph after the two
shape-subgraph transforms are exercised, so that a transform that changed an
answer would show up here rather than only as a different node count.

Nothing is claimed that is not measured: each shape's error is recorded, and a
stage that fails records the failure and lets the others carry on.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from unittest import mock

import numpy as np
import onnx
import onnxruntime as ort

from pyinfinitensor import onnx as frontend
from pyinfinitensor.onnx import OnnxStub, backend

parser = argparse.ArgumentParser()
parser.add_argument("--models", required=True, help="directory holding the .onnx files")
parser.add_argument("--out", required=True)
parser.add_argument("--rtol", type=float, default=1e-4)
parser.add_argument("--atol", type=float, default=1e-5)
args = parser.parse_args()

models_dir = Path(args.models)
out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

# Shapes go small -> large -> small and end where they started, so that growing
# and shrinking are both covered and the last inference reuses the first shape.
CASES = [
    (
        "attention",
        "attention.onnx",
        "x",
        [(2, 5, 32), (1, 1, 32), (2, 7, 32), (3, 16, 32), (1, 3, 32), (2, 5, 32)],
    ),
    (
        "resnet18",
        "resnet18.onnx",
        "images",
        [
            (2, 3, 48, 48),
            (1, 3, 32, 32),
            (2, 3, 64, 64),
            (3, 3, 48, 48),
            (1, 3, 96, 96),
            (2, 3, 48, 48),
        ],
    ),
]


def ort_session(raw):
    return ort.InferenceSession(raw, providers=["CPUExecutionProvider"])


def build_stub(raw, simplified):
    model = onnx.load_model_from_string(raw)
    if simplified:
        return OnnxStub(model, backend.cuda_runtime())
    with mock.patch.object(frontend, "simplify", side_effect=lambda m: (m, False)):
        return OnnxStub(model, backend.cuda_runtime())


def run_shapes(stub, session, input_name, shapes, rng):
    """Hand one stub every shape in turn and compare each output with ORT."""
    per_shape = []
    worst = 0.0
    for shape in shapes:
        x = rng.standard_normal(size=shape, dtype=np.float32)
        stub.set_input([list(shape)])
        stub.inputs[input_name].copyin_numpy(x)
        stub.run()

        reference = session.run(None, {input_name: x})
        names = list(stub.outputs.keys())
        entry = {"shape": list(shape), "outputs": {}}
        for name, want in zip(names, reference):
            got = stub.outputs[name].copyout_numpy().reshape(want.shape)
            err = float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64))))
            worst = max(worst, err)
            entry["outputs"][name] = {
                "shape": list(want.shape),
                "max_abs_error": err,
                "allclose": bool(
                    np.allclose(got, want, rtol=args.rtol, atol=args.atol)
                ),
            }
        per_shape.append(entry)
    return per_shape, worst


records = []
for name, filename, input_name, shapes in CASES:
    path = models_dir / filename
    if not path.exists():
        records.append({"model": name, "ok": False, "error": f"missing {path}"})
        continue
    raw = path.read_bytes()
    session = ort_session(raw)

    for simplified in (True, False):
        label = f"{name}/{'simplified' if simplified else 'unsimplified'}"
        record = {
            "model": name,
            "simplified": simplified,
            "device": "cuda",
            "shapes_requested": [list(s) for s in shapes],
        }
        try:
            rng = np.random.default_rng(20260916)
            stub = build_stub(raw, simplified)
            record["operators_initial"] = stub.handler.operator_count()
            record["shape_subgraph_initial"] = stub.shape_subgraph_size()

            # Plain graph first: this is the claim that matters for direction 3.
            per_shape, worst = run_shapes(stub, session, input_name, shapes, rng)
            record["plain"] = {
                "per_shape": per_shape,
                "max_abs_error": worst,
                "all_allclose": all(
                    o["allclose"] for e in per_shape for o in e["outputs"].values()
                ),
            }

            # Then the transforms, on a fresh stub so the plain result stands on
            # its own. A transform that broke an answer shows up as a mismatch
            # here, not merely as a different node count.
            rng2 = np.random.default_rng(20260916)
            stub2 = build_stub(raw, simplified)
            folded = stub2.fold_shape_subgraph()
            merged = stub2.merge_duplicate_shape_operators()
            record["transforms"] = {
                "folded_away": folded,
                "merged_away": merged,
                "operators_after": stub2.handler.operator_count(),
                "shape_subgraph_after": stub2.shape_subgraph_size(),
                "second_fold": stub2.fold_shape_subgraph(),
                "second_merge": stub2.merge_duplicate_shape_operators(),
            }
            per_shape2, worst2 = run_shapes(stub2, session, input_name, shapes, rng2)
            record["transformed"] = {
                "per_shape": per_shape2,
                "max_abs_error": worst2,
                "all_allclose": all(
                    o["allclose"] for e in per_shape2 for o in e["outputs"].values()
                ),
            }
            record["ok"] = (
                record["plain"]["all_allclose"] and record["transformed"]["all_allclose"]
            )
        except Exception as exc:  # noqa: BLE001 - the failure itself is the result
            record["ok"] = False
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback_tail"] = traceback.format_exc().strip().splitlines()[-1]

        records.append(record)
        summary = {
            k: record.get(k)
            for k in ("model", "simplified", "ok", "operators_initial", "error")
        }
        if "plain" in record:
            summary["plain_max_abs_error"] = record["plain"]["max_abs_error"]
        if "transforms" in record:
            summary["folded_away"] = record["transforms"]["folded_away"]
            summary["merged_away"] = record["transforms"]["merged_away"]
            summary["operators_after"] = record["transforms"]["operators_after"]
        if "transformed" in record:
            summary["transformed_max_abs_error"] = record["transformed"]["max_abs_error"]
        print("D3 " + json.dumps(summary), flush=True)

(out_dir / "direction3.json").write_text(json.dumps(records, indent=2) + "\n")

failed = [r for r in records if not r.get("ok")]
print(f"D3_COMPLETE configurations={len(records)} failed={len(failed)}", flush=True)
sys.exit(1 if failed else 0)
