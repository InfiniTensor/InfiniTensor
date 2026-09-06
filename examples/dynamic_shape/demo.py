"""Run one loaded model through five shapes, comparing every output to ORT."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import onnx
import onnxruntime as ort
from pyinfinitensor import backend
from pyinfinitensor.onnx import OnnxStub

from models import batch_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--model", type=Path, help="Use the exported digits CNN")
    parser.add_argument(
        "--output", type=Path, default=Path("dynamic_shape_results.json")
    )
    args = parser.parse_args()
    if args.cuda_graph and args.device != "cuda":
        parser.error("--cuda-graph requires --device cuda")
    model = onnx.load(args.model) if args.model else batch_model()
    runtime = backend.cpu_runtime() if args.device == "cpu" else backend.cuda_runtime()
    stub = OnnxStub(model, runtime)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    name = session.get_inputs()[0].name
    shapes = (
        [(1, 1, 8, 8), (2, 1, 12, 10), (4, 1, 16, 16), (3, 1, 10, 12), (1, 1, 8, 8)]
        if args.model
        else [(b, 2, 3) for b in (1, 2, 8, 3, 1)]
    )
    rng = np.random.default_rng(2026)
    records = []
    # These objects are intentionally constructed once, outside the loop.
    for shape in shapes:
        x = rng.normal(size=shape).astype(np.float32)
        expected = session.run(None, {name: x})
        started = time.perf_counter()
        actual = stub.infer({name: x}, cuda_graph=args.cuda_graph)
        elapsed = (time.perf_counter() - started) * 1000
        errors, output_shapes = {}, {}
        for output, reference in zip(session.get_outputs(), expected):
            result = actual[output.name]
            assert result.shape == reference.shape
            np.testing.assert_allclose(result, reference, rtol=1e-4, atol=1e-5)
            errors[output.name] = float(np.max(np.abs(result - reference), initial=0))
            output_shapes[output.name] = list(result.shape)
        record = {
            "input_shape": list(shape),
            "output_shapes": output_shapes,
            "max_abs_error": errors,
            "end_to_end_ms": elapsed,
            "memory": stub.memory_stats(),
        }
        if args.cuda_graph:
            record["captures"] = runtime.cuda_graph_capture_count()
        records.append(record)
        print(json.dumps(record))
    result = {
        "device": args.device,
        "cuda_graph": args.cuda_graph,
        "model": str(args.model) if args.model else "batch_model",
        "rtol": 1e-4,
        "atol": 1e-5,
        "onnxruntime": ort.__version__,
        "shape_optimization": stub.shape_optimization,
        "runs": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
