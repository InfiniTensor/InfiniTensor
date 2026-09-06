"""Plot measured JSON artifacts for the project report (no new benchmarks)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1] / "docs/dynamic_shape/results"
    benchmarks = [
        json.loads((root / f"benchmark_{name}.json").read_text())
        for name in ["cpu", "cuda", "cudagraph"]
    ]
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    x = np.arange(3)
    for offset, key, label, color in [
        (-0.18, 0, "Before", "#9BAEB8"),
        (0.18, -1, "Fold + reuse", "#187C92"),
    ]:
        values = [
            b["results"][key]["timings"]["end_to_end_ms"]["mean"] for b in benchmarks
        ]
        bars = axes[0].bar(x + offset, values, 0.36, label=label, color=color)
        axes[0].bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axes[0].set_xticks(x, ["CPU", "CUDA", "CUDA Graph"])
    axes[0].set_ylabel("Mean end-to-end latency (ms)")
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.2)
    axes[0].legend(frameon=False, fontsize=8)
    allocations = [
        benchmarks[0]["results"][i]["timed_pool_allocations"] for i in [0, -1]
    ]
    bars = axes[1].bar(
        ["Before", "Fold + reuse"], allocations, color=["#9BAEB8", "#187C92"], width=0.5
    )
    axes[1].bar_label(bars, padding=4)
    axes[1].set_ylim(0, max(allocations) * 1.2)
    axes[1].set_ylabel("Pool allocations / 100 timed runs")
    records = json.loads((root / "cpu_digits.json").read_text())["runs"]
    for field, label, color in [
        ("activation_required", "Logical requirement", "#9BAEB8"),
        ("activation_capacity", "Retained capacity", "#187C92"),
    ]:
        axes[2].plot(
            range(1, 6),
            [r["memory"][field] / 1024 for r in records],
            "o-",
            label=label,
            color=color,
        )
    axes[2].set_xticks(range(1, 6))
    axes[2].set_xlabel("Consecutive dynamic H/W run")
    axes[2].set_ylabel("CPU CNN activation pool (KiB)")
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    fig.tight_layout(w_pad=2)
    fig.savefig(root / "performance.png", bbox_inches="tight")
    fig.savefig(root / "performance.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
