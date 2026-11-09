# Validation on the supplied server

Validated on 2026-11-02 against branch base c06821eb97214f6ef989f7dcbf7a5b0c3572063b.
Both paths process all 4,992 raw MAT test samples, in ordered chunks of 64.

| Check | CPU reference | QEMU ordinary RISC-V floating point |
|---|---:|---:|
| Counts vs supplied reservoir counts | Exact, all 564,096 values | Exact, all 564,096 values |
| 120-dimensional features vs supplied inputs | Exact | Exact |
| Prediction max absolute error vs PyTorch checkpoint | 4.253862215364279e-6 | 4.253862215364279e-6 |
| Mean NMSE | 0.01696654296293974 | 0.016966542963236624 |
| Torch imported during inference | No | No |

Prediction max absolute difference between CPU and QEMU: 2.8359081412077103e-6.
CPU state is checked against independent NumPy at every chunk boundary; final
CPU/QEMU states agree within rtol=1e-12, atol=1e-8. Both paths produce exactly the
same counts and float32 features. QEMU audit records 78 successful graph sessions,
each serving MotionFeatures then MotionReadout. CPU validation launches no QEMU.
The main audit then contains synthetic boundary tests, including an intentionally
failed invalid-scale request; those entries are outside the 78 full-data sessions.

Additional checks pass for negative/zero inputs, negative fractional currents,
floating refractory boundaries, split state carry, invalid scales in both modes,
nonfinite state in both modes, and invalid weight shape. Existing LSM/ReLU MLP
validation also passes (all eight cases plus two original material samples), as
do test_graph, test_conv and Python API tests (three tests, one skipped).

Server artifacts are in /home/zhushuang/workspace/leinao/validation/motion_qemu_runner:
full/summary.json, full/reference_outputs.npz, full/qemu_outputs.npz,
full/qemu_audit.jsonl, model/, and logs/. Extra error-case audit is separate in
logs/additional_checks_audit.jsonl. Export's scipy dependency is isolated in deps/.
The tested build is /home/zhushuang/workspace/leinao/validation/lsm_mlp_qemu_runner/build.

Reproduce using that existing build:

```bash
cd /home/zhushuang/workspace/leinao/InfiniTensor
MOTION_QEMU_BUILD=/home/zhushuang/workspace/leinao/validation/lsm_mlp_qemu_runner/build \\
  bash scripts/reproduce_motion_qemu.sh
```

The NMSE uses previous standardized ground truth as explicit feedback, matching
the supplied model. This is a floating-point QEMU implementation; RVNE instruction
quantization and autonomous closed-loop control have separate acceptance criteria.
Reported wall times include validation overhead (CPU also runs the NumPy oracle)
and are not a CPU-vs-hardware performance benchmark.
