#!/usr/bin/env python3
"""Validate the hardware-kernel migration inventory without requiring hardware."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

VALID_STATUSES = {
    "infiniops_available",
    "runtime_sdk",
    "infiniccl_available",
    "upstream_todo",
}


def load_baseline(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        device, op = line.split(maxsplit=1)
        pairs.append((device, op))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--report-todo", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "docs/infiniops_kernel_coverage.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    baseline = load_baseline(root / manifest["baseline_file"])
    baseline_set = set(baseline)
    errors: list[str] = []

    overrides: dict[tuple[str, str], dict] = {}
    for entry in manifest["overrides"]:
        key = (entry["device"], entry["op"])
        if key not in baseline_set:
            errors.append(f"override is not in the hardware baseline: {key}")
        if key in overrides:
            errors.append(f"duplicate override: {key}")
        overrides[key] = entry

    resolved: list[dict] = []
    for device, op in baseline:
        entry = dict(manifest["default"])
        entry.update(manifest.get("device_defaults", {}).get(device, {}))
        entry.update(overrides.get((device, op), {}))
        entry["device"] = device
        entry["op"] = op
        if entry.get("status") not in VALID_STATUSES:
            errors.append(f"invalid status for {(device, op)}: {entry.get('status')}")
        if entry["status"] == "upstream_todo" and not entry.get("upstream"):
            errors.append(f"TODO without upstream target: {(device, op)}")
        if entry["status"] != "upstream_todo":
            source = root / entry.get("source", "")
            marker = entry.get("registration", "")
            if not source.is_file():
                errors.append(f"missing source for {(device, op)}: {source}")
            elif marker not in source.read_text(encoding="utf-8"):
                errors.append(f"missing registration marker for {(device, op)}: {marker}")
        resolved.append(entry)

    counts = Counter(entry["status"] for entry in resolved)
    print(f"baseline={len(resolved)} " + " ".join(
        f"{status}={counts[status]}" for status in sorted(VALID_STATUSES)))

    if args.report_todo:
        for entry in resolved:
            if entry["status"] == "upstream_todo":
                print(f"{entry['device']} {entry['op']} -> {entry['upstream']}: {entry['reason']}")

    if errors:
        print("coverage audit failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
