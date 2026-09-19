"""What a series of shapes costs, with the capacity reuse and without it.

A graph that keeps changing shape has to lay its activations out again each
time. Whether it also has to *allocate* them again is the question this
measures: the allocator holds activation storage at the high watermark of the
shapes it has seen, so a later shape that fits pays nothing. Trimming after
each inference gives up that slack, and is the comparison here -- it is the
policy of allocating exactly what the current shape needs, which is what a
graph without the reuse would do.

Four things are reported, for each policy:

  * how many times activation storage was asked of the runtime,
  * how long each inference took, split into laying out and running,
  * how many bytes were held at the peak, by the allocator and by the process,
  * and the throughput over the whole series.

Usage:
    python dynamic_shape_benchmark.py model.onnx --shapes 1x8 1x16 1x32
    python dynamic_shape_benchmark.py model.onnx --sweep batch=1,2,4,8

The shapes are given for the first input; a model with several inputs takes
`--shapes` once per input, comma-separated. Without either flag the model's own
declared shape is used with each symbolic dimension swept over a few sizes.
"""

import argparse
import statistics
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx

from pyinfinitensor.onnx import OnnxStub, backend


def process_peak_kib() -> Optional[int]:
    """The high watermark of the whole process, or None where unavailable.

    This is coarser than the allocator's own count -- it includes the
    interpreter, numpy, and the model file -- but it is the number that
    answers "did this actually cost the machine anything", and on a device
    the allocator's figure is the only one that means much anyway.
    """
    try:
        with open("/proc/self/status", "r") as status:
            for line in status:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def declared_shapes(model: onnx.ModelProto) -> List[List[object]]:
    """Each input's shape as the model declares it, symbols left as strings."""
    shapes = []
    for value in model.graph.input:
        dims = []
        for dim in value.type.tensor_type.shape.dim:
            dims.append(dim.dim_param if dim.dim_param else dim.dim_value)
        shapes.append(dims)
    return shapes


def parse_shape(text: str) -> List[int]:
    try:
        return [int(part) for part in text.replace("x", ",").split(",") if part]
    except ValueError:
        raise SystemExit('cannot read "{}" as a shape; try 1x3x224x224'.format(text))


def sweep_shapes(
    model: onnx.ModelProto, assignments: Dict[str, Sequence[int]]
) -> List[List[List[int]]]:
    """Fill each symbolic dimension in turn from the sizes given for it.

    One series is produced per size, every symbol taking that size's position;
    a symbol with no sizes given keeps whatever the model declared, and a
    symbol the model left at zero is given 1, which is what the importer does.
    """
    declared = declared_shapes(model)
    lengths = {len(sizes) for sizes in assignments.values()}
    if not lengths:
        raise SystemExit("--sweep needs at least one symbol=sizes assignment")
    if len(lengths) != 1:
        raise SystemExit("every swept symbol needs the same number of sizes")
    steps = lengths.pop()

    series = []
    for step in range(steps):
        shapes = []
        for dims in declared:
            resolved = []
            for dim in dims:
                if isinstance(dim, str):
                    sizes = assignments.get(dim)
                    resolved.append(int(sizes[step]) if sizes else 1)
                else:
                    resolved.append(dim if dim > 0 else 1)
            shapes.append(resolved)
        series.append(shapes)
    return series


class Measurement:
    """What one policy cost over one series of shapes."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.allocations = 0
        self.layout_ms: List[float] = []
        self.run_ms: List[float] = []
        self.peak_held = 0
        self.held: List[int] = []
        self.process_peak_kib: Optional[int] = None
        self.outputs: List[np.ndarray] = []

    @property
    def layout_run_ms(self) -> float:
        """Layout (including trim) plus execution; excludes data copies."""
        return sum(self.layout_ms) + sum(self.run_ms)

    def summarise(self) -> str:
        inferences = len(self.run_ms)
        held = "{:.2f} MiB".format(self.peak_held / (1024 * 1024))
        process = (
            "{:.1f} MiB".format(self.process_peak_kib / 1024)
            if self.process_peak_kib is not None
            else "n/a"
        )
        return "\n".join(
            [
                "  {}".format(self.label),
                "    allocations, steady state: {}".format(self.allocations),
                "    layout, median          : {:.3f} ms".format(
                    statistics.median(self.layout_ms) if self.layout_ms else 0.0
                ),
                "    run, median             : {:.3f} ms".format(
                    statistics.median(self.run_ms) if self.run_ms else 0.0
                ),
                "    layout+run, {:<3} calls   : {:.1f} ms (excludes copies)".format(
                    inferences, self.layout_run_ms
                ),
                "    layout+run rate         : {:.1f} inferences/s".format(
                    inferences / (self.layout_run_ms / 1000)
                    if self.layout_run_ms
                    else 0.0
                ),
                "    peak held by allocator  : {}".format(held),
                "    mean held across series : {:.2f} MiB".format(
                    statistics.mean(self.held) / (1024 * 1024) if self.held else 0.0
                ),
                "    peak held by process    : {}".format(process),
            ]
        )


def measure(
    model: onnx.ModelProto,
    runtime,
    series: Sequence[Sequence[Sequence[int]]],
    inputs: Sequence[Sequence[np.ndarray]],
    label: str,
    trim_each_time: bool,
) -> Measurement:
    """Drive `series` through a fresh graph and record what it cost.

    With `trim_each_time` the slack is handed back after every inference, so
    each shape has to allocate storage of exactly its own size -- the policy
    the reuse is being compared against.
    """
    result = Measurement(label)
    stub = OnnxStub(model, runtime)
    stub.init()
    input_names = list(stub.inputs)

    def inference(shapes: Sequence[Sequence[int]], data: Sequence[np.ndarray]):
        """One inference, timed in two halves, with the policy's own cost.

        Handing the slack back belongs to the layout half rather than after it:
        it is the cost of the policy, and leaving it outside the timing would
        credit the exact-fit policy with work it really does. What the next
        `set_input` then has to allocate is charged to the next inference,
        which is where a deployment would feel it too.
        """
        start = time.perf_counter()
        stub.set_input([list(shape) for shape in shapes])
        if trim_each_time:
            stub.trim_memory()
        layout = (time.perf_counter() - start) * 1000

        for name, array in zip(input_names, data):
            stub.tensors[name].copyin_numpy(array)

        start = time.perf_counter()
        stub.run()
        return layout, (time.perf_counter() - start) * 1000

    # First touch of freshly allocated pages, kernel selection and a cold cache
    # all land on whichever policy is measured first, and they are worth more
    # than everything being measured here: the first inference of this model
    # ran forty times slower than the twentieth. So each policy warms up on its
    # own series, untimed, and the counters are read only afterwards.
    for shapes, data in zip(series, inputs):
        inference(shapes, data)
    warmed = stub.activation_allocations()

    for shapes, data in zip(series, inputs):
        layout, run = inference(shapes, data)
        result.layout_ms.append(layout)
        result.run_ms.append(run)
        held = stub.allocated_bytes()
        result.peak_held = max(result.peak_held, held)
        result.held.append(held)
        result.outputs.append(next(iter(stub.outputs.values())).copyout_numpy())

    # Allocations during the warmup are the ones a first run has to make
    # whatever the policy, so what is reported is the steady state: what this
    # policy costs to keep going, once nothing is new.
    result.allocations = stub.activation_allocations() - warmed
    result.process_peak_kib = process_peak_kib()
    return result


def representative(trials: Sequence[Measurement]) -> Measurement:
    """The middle trial of several, by how long an inference took in it.

    A single pass gives a total that is dominated by a handful of slow
    inferences, and which of the two policies collects them depends on which
    ran first. Taking the trial with the median inference time throws that
    away and keeps a real measurement rather than an average of several -- the
    latencies reported are ones that actually happened together.

    The allocation counts are the same in every trial, being counted rather
    than timed, so nothing is lost there.
    """
    if not trials:
        raise SystemExit("no trials to summarise")
    ordered = sorted(
        trials,
        key=lambda trial: statistics.median(trial.run_ms) if trial.run_ms else 0.0,
    )
    return ordered[len(ordered) // 2]


def compare_outputs(left: Measurement, right: Measurement) -> Tuple[bool, float]:
    """Whether the two policies agreed, and by how much they did not.

    A reuse that answers differently is not an optimisation, so this is the
    part of the measurement that has to hold before any of the rest counts.
    """
    worst = 0.0
    for a, b in zip(left.outputs, right.outputs):
        if a.shape != b.shape:
            return False, float("inf")
        if a.size == 0:
            continue
        worst = max(worst, float(np.max(np.abs(a.astype(np.float64) - b))))
    return worst == 0.0, worst


def build_series(
    model: onnx.ModelProto, args: argparse.Namespace
) -> List[List[List[int]]]:
    if args.shapes:
        return [[parse_shape(text) for text in step.split(":")] for step in args.shapes]
    if args.sweep:
        assignments: Dict[str, Sequence[int]] = {}
        for item in args.sweep:
            if "=" not in item:
                raise SystemExit('--sweep takes symbol=sizes, as in "batch=1,2,4"')
            symbol, sizes = item.split("=", 1)
            assignments[symbol] = parse_shape(sizes)
        return sweep_shapes(model, assignments)

    # Nothing asked for: sweep every symbol the model declares over a few sizes.
    symbols = {
        dim for dims in declared_shapes(model) for dim in dims if isinstance(dim, str)
    }
    if not symbols:
        raise SystemExit(
            "this model declares no symbolic dimension, so there is no shape "
            "series to measure; pass --shapes to give one anyway"
        )
    sizes = [1, 2, 4, 8, 4, 2, 1, 8]
    return sweep_shapes(model, {symbol: sizes for symbol in symbols})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure what a series of shapes costs, with the activation "
        "capacity reuse and against allocating exactly what each shape needs."
    )
    parser.add_argument("model", help="path to an ONNX model")
    parser.add_argument(
        "--shapes",
        nargs="+",
        help="one shape per step, as 1x3x224x224; several inputs are separated "
        "by colons within a step",
    )
    parser.add_argument(
        "--sweep",
        nargs="+",
        help="symbol=sizes, as batch=1,2,4,8; every symbol needs the same "
        "number of sizes",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=4,
        help="how many times to go round the series (default 4)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="how many times to measure each policy, taking turns (default 5); "
        "one pass each is not enough, the order decides the answer",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="seed for the input data (default 0)"
    )
    args = parser.parse_args()

    model = onnx.load(args.model)
    series = build_series(model, args) * max(1, args.repeat)

    # The same data through both policies, so the outputs can be compared. It
    # is generated once and reused, rather than per policy, because a policy
    # that got different numbers would be compared against nothing.
    rng = np.random.default_rng(args.seed)
    inputs = [
        [rng.standard_normal(shape).astype(np.float32) for shape in shapes]
        for shapes in series
    ]

    print("model     : {}".format(args.model))
    print("operators : {}".format(len(model.graph.node)))
    print("declared  : {}".format(declared_shapes(model)))
    print(
        "series    : {} inferences over {} distinct shapes".format(
            len(series), len({tuple(map(tuple, shapes)) for shapes in series})
        )
    )
    print()

    runtime = backend.cpu_runtime()

    # Whichever policy is measured first pays for the process as a whole: the
    # first inference of a 51-operator model here ran forty times slower than
    # the twentieth, and that is worth more than the difference being looked
    # for. So one throwaway pass warms everything shared before anything is
    # recorded, and then the two policies take turns, because a single pass each
    # gave answers that swapped when the order did.
    measure(model, runtime, series, inputs, "warmup", False)

    reuse_trials, naive_trials = [], []
    for trial in range(args.trials):
        order = [
            (reuse_trials, "capacity reuse (default)", False),
            (naive_trials, "exact fit (trim each time)", True),
        ]
        if trial % 2:
            order.reverse()
        for collected, label, trim in order:
            collected.append(measure(model, runtime, series, inputs, label, trim))

    reuse = representative(reuse_trials)
    naive = representative(naive_trials)

    print("trials    : {}, policies alternating".format(args.trials))
    print()
    print(reuse.summarise())
    print()
    print(naive.summarise())
    print()

    agreed, worst = compare_outputs(reuse, naive)
    print("outputs agree exactly : {}".format(agreed))
    if not agreed:
        print("worst difference      : {}".format(worst))
        raise SystemExit(
            "the two policies answered differently, so the measurement above "
            "says nothing about an optimisation"
        )

    if naive.allocations:
        print(
            "allocations           : {} against {}, {:.1f}x fewer".format(
                reuse.allocations,
                naive.allocations,
                naive.allocations / max(reuse.allocations, 1),
            )
        )
    # Totals are not compared across policies: they are sums, so a handful of
    # slow inferences decides them, and which policy collects those depends on
    # the order. The median of an inference is what survives that.
    reuse_median = statistics.median(reuse.layout_ms) + statistics.median(reuse.run_ms)
    naive_median = statistics.median(naive.layout_ms) + statistics.median(naive.run_ms)
    if naive_median:
        drift = (reuse_median - naive_median) / naive_median * 100
        print(
            "sum of layout/run medians (excludes copies): {:.3f} ms against {:.3f} ms, {:+.1f}%{}".format(
                reuse_median,
                naive_median,
                drift,
                "  (within trial-to-trial spread)" if abs(drift) < 10 else "",
            )
        )
    print(
        "peak held             : {:.2f} MiB against {:.2f} MiB, {:+.1f}%".format(
            reuse.peak_held / (1024 * 1024),
            naive.peak_held / (1024 * 1024),
            (
                (reuse.peak_held - naive.peak_held) / naive.peak_held * 100
                if naive.peak_held
                else 0.0
            ),
        )
    )
    # The peak is the same by construction whenever the series contains its own
    # largest shape, since both policies have to hold it at that moment. What
    # the reuse costs is what it goes on holding afterwards, so the mean across
    # the series is the figure that shows the trade being made.
    reuse_mean = statistics.mean(reuse.held) if reuse.held else 0.0
    naive_mean = statistics.mean(naive.held) if naive.held else 0.0
    print(
        "mean held             : {:.2f} MiB against {:.2f} MiB, {:+.1f}%".format(
            reuse_mean / (1024 * 1024),
            naive_mean / (1024 * 1024),
            (reuse_mean - naive_mean) / naive_mean * 100 if naive_mean else 0.0,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
