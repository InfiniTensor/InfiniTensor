#!/usr/bin/env python3
"""Fail early when a Python package cannot create its requested provider."""
import argparse
import importlib
import json
import os
import platform
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-modules", default="", help="comma-separated modules loaded before backend")
    parser.add_argument("--backend", default="", help="legacy alias for the provider device")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--require-infini", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--model-dir", default="", help="optional local ONNX embedding package for capability self-check")
    parser.add_argument("--input-shape", default="1,17", help="batch,sequence used with --model-dir")
    args = parser.parse_args()
    modules = [item.strip() for item in args.provider_modules.split(",") if item.strip()]
    if modules:
        os.environ["INFINIOPS_PROVIDER_MODULES"] = ",".join(modules)
    report = {"python": sys.version.split()[0], "platform": platform.platform(), "provider_modules": modules}
    try:
        for module in modules:
            importlib.import_module(module)
        from pyinfinitensor import backend
        report["backend"] = str(backend)
        device = args.backend or ("nvidia" if args.require_infini else "cpu")
        runtime = backend.runtime(device, args.device)
        report.update({"device": device, "device_id": args.device, "runtime": type(runtime).__name__})
        if args.model_dir:
            import onnx
            from pyinfinitensor.onnx import OnnxStub
            dims = tuple(int(value) for value in args.input_shape.split(","))
            if len(dims) != 2 or any(value < 1 for value in dims):
                raise ValueError("--input-shape must be two positive integers: batch,sequence")
            stub = OnnxStub(onnx.load(os.path.join(args.model_dir, "model.onnx")), runtime)
            shapes = {name: dims for name in ("input_ids", "attention_mask", "token_type_ids")}
            report["capabilities"] = stub.check_capabilities(shapes)
    except Exception as error:
        report["error"] = "{}: {}".format(type(error).__name__, error)
        print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else report["error"], file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else "InfiniTensor install self-check passed: {}".format(report))


if __name__ == "__main__":
    main()
