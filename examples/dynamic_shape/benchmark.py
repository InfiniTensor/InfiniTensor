"""Compare existing pooled capacity reuse against explicit trim-after-run.

Timings include shape planning, allocation, copies, kernels and (when enabled)
trim. This measures the complete frontend call, not isolated kernel latency.
"""

import json
import statistics
import time
from pathlib import Path
import numpy as np
from demo import model
from pyinfinitensor import backend
from pyinfinitensor.onnx import DynamicOnnxStub


def measure(trim, repeats=20):
    stub = DynamicOnnxStub(model(), backend.cpu_runtime())
    inputs = [np.full((b, 3, 16, 16), b, np.float32) for b in [8, 4, 2, 1, 8]]
    stub.run({"x": inputs[0]})
    start_count = stub.handler.memory_stats()["activation_allocations"]
    times, capacities, required = [], [], []
    for _ in range(repeats):
        for x in inputs:
            start = time.perf_counter_ns()
            result = stub.run({"x": x})["y"]
            before_trim = stub.handler.memory_stats()
            if trim:
                stub.handler.trim_memory()
            times.append((time.perf_counter_ns() - start) / 1e6)
            np.testing.assert_allclose(result, x.reshape(x.shape[0], -1) + 0.25)
            capacities.append(before_trim["activation_capacity_bytes"])
            required.append(before_trim["activation_required_bytes"])
    stats = stub.handler.memory_stats()
    return {
        "measured_runs": len(times),
        "activation_allocations_after_warmup": stats["activation_allocations"]
        - start_count,
        "median_end_to_end_ms": statistics.median(times),
        "p95_end_to_end_ms": float(np.percentile(times, 95)),
        "peak_committed_activation_capacity_bytes": max(capacities),
        "peak_required_activation_bytes": max(required),
        "final_memory_stats": stats,
    }


if __name__ == "__main__":
    result = {"capacity_reuse": measure(False), "trim_after_each_run": measure(True)}
    path = Path("artifacts/dynamic_shape")
    path.mkdir(parents=True, exist_ok=True)
    (path / "memory_benchmark.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
