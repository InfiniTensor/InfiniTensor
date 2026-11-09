# Baxter motion control: complete CPU and QEMU paths

This example extends the LSM/QEMU bridge with the delivered motion-control model.
Both graphs start with raw 21-dimensional MAT inputs and explicit state, execute
delta encoding, 50 reservoir steps, count standardization, previous-feedback
concatenation, a 120->128->7 Tanh readout, and output denormalization.

`MotionFeatures` uses double weights and state. `MotionReadout` uses float32
Linear/Tanh/Linear and double denormalization. CPU execution and the RISC-V runner
compile the same portable calculation code. The independent NumPy reservoir and
checkpoint evaluation provide external references. QEMU runs ordinary RISC-V
floating-point code; this example does not quantize the model for RVNE instructions.

Run on the supplied server (Torch and scipy are required for export only):

```bash
bash scripts/reproduce_motion_qemu.sh
```

The script accepts `MOTION_MATERIALS`, `MOTION_QEMU_ARTIFACTS`, `MOTION_QEMU_BUILD`,
`RVNE_SDK`, `MOTION_CHUNK` (default 64), `MOTION_PYTHON_DEPS` (default
`$MOTION_QEMU_ARTIFACTS/deps`) and optional `MOTION_SAMPLES`.
Install export dependencies in the chosen environment or supply a separate
dependency directory through `PYTHONPATH`; the script does not install packages.
For example, `python -m pip install --target /path/to/artifacts/deps --no-deps scipy==1.15.3`
provides scipy without modifying the conda environment.
The default environment activation follows the existing server reproduction script.

For inference after export, configure `PYTHONPATH` with the build directory and
`pyinfinitensor/src`. QEMU also needs `INFINI_RVNE_QEMU`, `INFINI_RVNE_SYSROOT`,
`INFINI_RVNE_RUNNER`, and optionally `INFINI_RVNE_TIMEOUT_MS`/`INFINI_RVNE_AUDIT`:

```bash
python examples/motion_qemu/infer.py --model-dir /path/to/model \
  --execution reference --check-oracle --output-dir /path/to/cpu
python examples/motion_qemu/infer.py --model-dir /path/to/model \
  --execution qemu --output-dir /path/to/qemu
```

Inference does not import Torch. Artifacts include both ONNX models, dataset,
manifest, CPU/QEMU outputs, validation summary and QEMU request audit. Inference
returns counts/features/state as well as physical-space predictions. Validation
checks the complete test set against supplied counts/features and checkpoint
outputs, checks CPU state against NumPy, compares both paths, and exercises
negative/zero inputs, negative current floor, refractory boundaries, split state
carry and invalid scales. CPU graphs must launch no QEMU; each QEMU graph run must
record one process serving `MotionFeatures` and `MotionReadout` successfully.

## Model contract v1

`MotionFeatures` inputs, in order: raw `[S,21]`, previous raw `[21]`, standardized
previous feedback `[S,7]`, state `[3,113]`, input weights `[42,113]`, recurrent
weights `[113,113]`, count mean `[113]`, count scale `[113]`; all are double.
Outputs: float32 features `[S,120]`, double state `[3,113]`, double last raw `[21]`,
int32 counts `[S,113]`. State rows are current, voltage, refractory. S is a
sequential sample dimension (1..1024). I/V/refractory carry between samples;
feedback spikes and counts reset per sample. Chunk boundaries occur only between
complete 50-step samples. Previous raw input and state explicitly connect chunks.

`MotionReadout` inputs: float32 features, W1 `[120,128]`, b1 `[128]`, W2 `[128,7]`,
b2 `[7]`, double output mean `[7]`, output scale `[7]`; output is double `[S,7]`.
Weights in this interface use source/target order. Shapes and finite values are
validated; scales must be positive. The custom domain is `com.infinitensor.rvne`
opset 1; `execution` is `reference` or `qemu`. Session protocol v2 adds commands
3/4 while retaining commands 1/2 for LSM/ReLU MLP; rebuild host and runner together.

The checkpoint's evaluation consumes previous standardized **ground truth**
(first row is standardized zero), supplied as explicit external feedback. The
reported NMSE reproduces that condition and does not describe autonomous control.
All normalization statistics come from training data. The exporter reads the
provided archives without calling the random reservoir-generation functions.
