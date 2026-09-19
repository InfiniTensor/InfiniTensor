"""What giving capacity back costs on a GPU, where the allocator is not cheap.

The tradeoff between holding activation capacity and handing it back was
measured on the CPU (B phase): three policies, no dominant one, `conditional`
buying most of `always`'s memory saving at a fraction of its latency. That
result rests on something specific to the host, though -- `malloc` is cheap and
does not synchronise. `cudaMalloc` does both the opposite: it is a synchronising
call an order or two more expensive, so the same shrink that costs a host
allocator a few microseconds costs the device far more.

So the CPU curve should not carry over, and this asks the same questions again
on the device to find out how it differs. The point is the comparison, so the
method is the CPU one unchanged -- same three policies, same four trajectories,
same separation between tuning and reporting sequences, same metric breakdown.
Only the runtime differs, which is what makes the two sets of numbers
comparable.

Two things are kept from the CPU script deliberately:

  * `sweep` does not run the model. A policy changes allocation, and a forward
    pass at ResNet18's cost would bury that difference under work no policy
    touches. On the device that matters more, not less.
  * Correctness is established separately, with runs, against onnxruntime. A
    policy that saved memory by handing back storage a live tensor points into
    would show there and in no timing.

Device memory is also read from the driver, because `allocated_bytes()` reports
what the allocator holds and says nothing about what was returned to the driver
-- on the device those are different questions, and the second one is why a
shrink is slow.
"""

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from pyinfinitensor.onnx import OnnxStub, backend

parser = argparse.ArgumentParser()
parser.add_argument("--models", required=True, help="directory holding the .onnx files")
parser.add_argument("--out", required=True, help="directory for the json record")
parser.add_argument("--passes", type=int, default=20)
parser.add_argument("--trials", type=int, default=12)
parser.add_argument("--warmup", type=int, default=3)
args = parser.parse_args()

models_dir = Path(args.models)
out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

clock = time.perf_counter_ns

POLICIES = ("never", "always", "conditional")


def device_used_bytes():
    """What the driver says is in use, or 0 when it cannot be read.

    `allocated_bytes()` is the allocator's own book-keeping. This is the other
    side of it: whether a shrink actually gave anything back to the driver.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True).stdout
        return int(out.strip().splitlines()[0]) * 1024 * 1024
    except Exception:  # noqa: BLE001 -- a missing reading is not a failure
        return 0


class Conditional:
    """Shrink once utilisation has stayed low, then hold off for a cooldown.

    Taken from the CPU experiment unchanged, so that a difference in the
    results is a difference in the device and not in the policy.
    """

    def __init__(self, threshold, patience, cooldown):
        self.threshold = threshold
        self.patience = patience
        self.cooldown = cooldown
        self.low = 0
        self.since = 10**9

    def reset(self):
        self.low = 0
        self.since = 10**9

    def should_trim(self, peak, capacity):
        self.since += 1
        if capacity == 0:
            return False
        self.low = self.low + 1 if peak <= self.threshold * capacity else 0
        if self.low >= self.patience and self.since >= self.cooldown:
            self.low = 0
            self.since = 0
            return True
        return False


def trajectories(large, mid, small, length=24):
    """The same four sequences the CPU result was reported on."""
    peak_then_small = [large] + [small] * (length - 1)
    alternating = [large if i % 2 == 0 else small for i in range(length)]
    stable = [mid] * length
    half = length // 2
    return_to_peak = [small] * (half - 1) + [large] + [small] * (length - half)
    return {
        "peak_then_small": peak_then_small,
        "alternating": alternating,
        "stable": stable,
        "return_to_peak": return_to_peak[:length],
    }


def tuning_trajectories(large, mid, small, seed, length=24):
    """Sequences the parameters are chosen on, kept apart from the four above."""
    rng = np.random.default_rng(seed)
    pool = [large, mid, small]
    drawn = {}
    for name, weights in (
        ("mostly_small", [0.1, 0.2, 0.7]),
        ("mostly_large", [0.6, 0.2, 0.2]),
        ("even", [1 / 3, 1 / 3, 1 / 3]),
    ):
        idx = rng.choice(len(pool), size=length, p=weights)
        drawn["tune_" + name] = [pool[i] for i in idx]
    return drawn


def sweep(stub, sequence, policy, controller, passes):
    """Drive one policy over one sequence and count what it costs.

    The model is not run, for the reason given at the top of the file.
    """
    if controller is not None:
        controller.reset()
    rows = []
    trims = 0
    trim_ns = 0
    set_input_ns = 0
    allocations_before = stub.activation_allocations()
    held = []
    transient = []
    for _ in range(passes):
        for shape in sequence:
            capacity_before = stub.activation_capacity()
            t = clock()
            stub.set_input([list(shape)])
            set_input_ns += clock() - t
            peak = stub.activation_peak()
            capacity = stub.activation_capacity()
            if capacity > capacity_before:
                # A larger storage is taken while the old one is still held.
                transient.append(capacity_before + capacity)
            trimmed = False
            if policy == "always":
                trimmed = True
            elif policy == "conditional":
                trimmed = controller.should_trim(peak, capacity)
            if trimmed:
                before = stub.activation_capacity()
                t = clock()
                stub.trim_memory()
                trim_ns += clock() - t
                trims += 1
                after = stub.activation_capacity()
                transient.append(before + after)
                capacity = after
            held.append(stub.allocated_bytes())
            rows.append({
                "shape": list(shape),
                "demand": peak,
                "capacity": capacity,
                "held": stub.allocated_bytes(),
                "trimmed": trimmed,
            })
    iterations = passes * len(sequence)
    return {
        "policy": policy,
        "iterations": iterations,
        "allocations": stub.activation_allocations() - allocations_before,
        "trims": trims,
        "set_input_ms_total": set_input_ns / 1e6,
        "trim_ms_total": trim_ns / 1e6,
        "per_shape_change_us": (set_input_ns + trim_ns) / iterations / 1e3,
        # Reported on its own as well as in the total: on the device this is
        # the term expected to dominate, which is the whole question here.
        "trim_us_each": (trim_ns / trims / 1e3) if trims else 0.0,
        "held_mean": statistics.mean(held),
        "held_peak": max(held),
        "transient_peak": max(transient) if transient else 0,
        "rows_sample": rows[: len(sequence)],
    }


CASES = [
    ("attention", "attention.onnx", "x", "y",
     (4, 32, 32), (2, 16, 32), (1, 4, 32)),
    ("resnet18", "resnet18.onnx", "images", "logits",
     (4, 3, 96, 96), (2, 3, 48, 48), (1, 3, 32, 32)),
]

results = {}
failed = 0

for name, filename, input_name, output_name, large, mid, small in CASES:
    path = models_dir / filename
    if not path.exists():
        print("CAP missing model: %s" % path)
        failed += 1
        continue

    raw = path.read_bytes()

    def make_stub(first_shape):
        return OnnxStub(
            onnx.load_model_from_string(raw),
            backend.cuda_runtime(),
            input_shapes={input_name: list(first_shape)},
        )

    def feed(shape):
        rng = np.random.default_rng(20260908)
        return rng.standard_normal(size=shape, dtype=np.float32)

    record = {"model": name, "ok": False, "error": None}
    try:
        # Correctness first, and with runs: handing back storage a live tensor
        # points into would show here and in no timing.
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        reference = ort.InferenceSession(
            raw, options, providers=["CPUExecutionProvider"])
        correctness = []
        for policy in POLICIES:
            stub = make_stub(mid)
            controller = Conditional(0.5, 2, 3) if policy == "conditional" else None
            worst = 0.0
            checks = 0
            for shape in [mid, large, small, small, large, mid, small]:
                stub.set_input([list(shape)])
                if policy == "always":
                    stub.trim_memory()
                elif policy == "conditional" and controller.should_trim(
                        stub.activation_peak(), stub.activation_capacity()):
                    stub.trim_memory()
                data = feed(shape)
                stub.inputs[input_name].copyin_numpy(data)
                stub.run()
                got = stub.outputs[output_name].copyout_numpy()
                want = reference.run([output_name], {input_name: data})[0]
                np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
                worst = max(worst, float(np.max(np.abs(got - want))))
                checks += 1
            correctness.append(
                {"policy": policy, "ort_checks": checks, "max_abs_error": worst})
            del stub
        record["correctness"] = correctness

        # Parameters are chosen on their own sequences and then left alone.
        candidates = [
            {"threshold": t, "patience": p, "cooldown": c}
            for t in (0.4, 0.5, 0.6)
            for p in (2, 3)
            for c in (3, 5)
        ]
        tuning = []
        tune_set = tuning_trajectories(large, mid, small, seed=20260914)
        for candidate in candidates:
            held_means = []
            costs = []
            for sequence in tune_set.values():
                stub = make_stub(mid)
                got = sweep(stub, sequence, "conditional",
                            Conditional(**candidate), args.warmup)
                held_means.append(got["held_mean"])
                costs.append(got["per_shape_change_us"])
                del stub
            tuning.append({
                **candidate,
                "held_mean": statistics.mean(held_means),
                "per_shape_change_us": statistics.mean(costs),
            })
        # Least held memory, and among ties the cheapest -- the rule stated
        # before the numbers are seen rather than picked to suit them.
        chosen = min(tuning,
                     key=lambda r: (r["held_mean"], r["per_shape_change_us"]))
        record["chosen_parameters"] = chosen
        record["tuning"] = tuning

        measured = {}
        for label, sequence in trajectories(large, mid, small).items():
            per_policy = {p: [] for p in POLICIES}
            for trial in range(args.trials):
                # Balanced order, so a policy never always follows the same one.
                order = [POLICIES[(trial + i) % len(POLICIES)]
                         for i in range(len(POLICIES))]
                for policy in order:
                    stub = make_stub(mid)
                    controller = (Conditional(chosen["threshold"],
                                              chosen["patience"],
                                              chosen["cooldown"])
                                  if policy == "conditional" else None)
                    sweep(stub, sequence, policy, controller, args.warmup)
                    got = sweep(stub, sequence, policy, controller, args.passes)
                    got["trial"] = trial
                    got["device_used"] = device_used_bytes()
                    per_policy[policy].append(got)
                    del stub

            summary = {}
            for policy, runs in per_policy.items():
                summary[policy] = {
                    "trials": len(runs),
                    "held_mean": statistics.mean(r["held_mean"] for r in runs),
                    "held_peak": max(r["held_peak"] for r in runs),
                    "transient_peak": max(r["transient_peak"] for r in runs),
                    "per_shape_change_us": statistics.median(
                        r["per_shape_change_us"] for r in runs),
                    "per_shape_change_us_mean": statistics.mean(
                        r["per_shape_change_us"] for r in runs),
                    "trim_us_each": statistics.median(
                        r["trim_us_each"] for r in runs),
                    "trims": runs[0]["trims"],
                    "allocations": runs[0]["allocations"],
                    "device_used_max": max(r["device_used"] for r in runs),
                }
            base = summary["never"]["per_shape_change_us"]
            for policy, row in summary.items():
                row["latency_ratio"] = (
                    round(row["per_shape_change_us"] / base, 3) if base else None)
                row["held_vs_never"] = (
                    round(row["held_mean"] / summary["never"]["held_mean"], 4)
                    if summary["never"]["held_mean"] else None)
            measured[label] = summary

        record["trajectories"] = measured
        record["ok"] = True
    except Exception as exc:  # noqa: BLE001 -- the message is the finding
        record["error"] = "%s: %s" % (type(exc).__name__, exc)
        failed += 1

    results[name] = record
    # One line per trajectory per policy, so the shape of the tradeoff is
    # readable without opening the json.
    if record["ok"]:
        # Every trajectory, not just the first. Naming one of them here once
        # meant three of the four were measured, written to the json, and never
        # looked at -- and the run still said it had completed.
        for label in record["trajectories"]:
            for policy in POLICIES:
                row = record["trajectories"][label][policy]
                print("CAP %-9s %-15s %-12s held=%.0f x%s  us=%.1f x%s  trim_us=%.1f"
                      % (name, label, policy, row["held_mean"],
                         row["held_vs_never"], row["per_shape_change_us"],
                         row["latency_ratio"], row["trim_us_each"]))
    else:
        print("CAP %-9s FAILED %s" % (name, record["error"]))

(out_dir / "capacity_cuda.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8")
print("CAP_COMPLETE models=%d failed=%d" % (len(CASES), failed))
