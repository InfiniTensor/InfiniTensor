#!/usr/bin/env python3
"""Run M1 benchmark cases in fresh processes with rotated engine order."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


CASES = ((1, 32), (1, 128), (1, 512), (8, 128), (32, 128))


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--accept-script", default=str(Path(__file__).with_name("accept_m1.py")))
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", choices=("nvidia", "cambricon", "ascend"), required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--execution-mode", choices=("normal", "auto", "graph"), default="auto")
    parser.add_argument("--compare-execution-modes", action="store_true",
                        help="measure normal/auto in the same rotated baseline matrix")
    args = parser.parse_args()
    output = Path(args.output)
    logs = output / "benchmark-logs"
    logs.mkdir(parents=True, exist_ok=True)
    engines = (["infinitensor-normal", "infinitensor-auto"] if args.compare_execution_modes
               else ["infinitensor"]) + ["pytorch"] + (["ort"] if args.device == "nvidia" else [])
    orders = [engines[index:] + engines[:index] for index in range(len(engines))]
    records = []
    for round_index in range(1, 4):
        order = orders[(round_index - 1) % len(orders)]
        for engine in order:
            actual_engine = "infinitensor" if engine.startswith("infinitensor") else engine
            mode = engine.split("-", 1)[1] if engine.startswith("infinitensor-") else args.execution_mode
            case_output = output / engine if args.compare_execution_modes else output
            case_output.mkdir(parents=True, exist_ok=True)
            cases = [("{}x{}".format(*case), case) for case in CASES] + [("mixed", None)]
            for case_name, case in cases:
                command = [args.python, args.accept_script, "--stage", "benchmark",
                           "--model-dir", args.model_dir, "--device", args.device,
                           "--device-id", str(args.device_id), "--output", str(case_output),
                           "--engine", actual_engine, "--benchmark-round", str(round_index),
                           "--warmup", str(args.warmup), "--runs", str(args.runs),
                           "--execution-mode", mode]
                if case is None:
                    command.append("--benchmark-mixed")
                else:
                    command.extend(("--benchmark-case", str(case[0]), str(case[1])))
                log_path = logs / "round-{}-{}-{}.log".format(round_index, engine, case_name)
                started = time.time()
                completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, env=os.environ.copy())
                log_path.write_text(completed.stdout, encoding="utf-8")
                report_path = case_output / "benchmark-{}-round-{}-{}.json".format(
                    actual_engine, round_index, case_name)
                records.append({"round": round_index, "engine": engine, "execution_mode": mode, "case": case_name,
                                "returncode": completed.returncode,
                                "elapsed_s": time.time() - started,
                                "report": str(report_path), "log": str(log_path),
                                "report_sha256": file_sha256(report_path) if report_path.is_file() else None})
    result = {"status": "PASS" if all(item["returncode"] == 0 for item in records) else "FAIL",
              "shared_environment": True, "engine_orders": orders, "records": records}
    (output / "benchmark-matrix.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
