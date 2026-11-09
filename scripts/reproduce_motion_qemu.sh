#!/usr/bin/env bash
# Complete raw MAT -> floating-point reservoir -> Tanh readout on host and QEMU.
set -eo pipefail
task_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
task_parent=$(dirname "$task_repo")
task_artifacts=${MOTION_QEMU_ARTIFACTS:-$task_parent/validation/motion_qemu_runner}
task_build=${MOTION_QEMU_BUILD:-$task_artifacts/build}
task_sdk=${RVNE_SDK:-$task_parent/validation/rvne_m0_20261012/sdk/rvne/toolchain}
task_materials=${MOTION_MATERIALS:-$task_parent/documents-export-2026-10-30}
task_deps=${MOTION_PYTHON_DEPS:-$task_artifacts/deps}
source /home/zhushuang/workspace/envs/set_a100_env.sh
conda activate infiniops_test
set -u
mkdir -p "$task_artifacts/logs" "$task_artifacts/bin"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
if [[ -d "$task_deps" ]]; then export PYTHONPATH="$task_deps${PYTHONPATH:+:$PYTHONPATH}"; fi
"$task_sdk/llvm/bin/clang++" --target=riscv64-unknown-linux-gnu \
  -std=c++17 -march=rv64imafdcvzne -mabi=lp64d -O2 -ffp-contract=off -fno-fast-math -fuse-ld=lld \
  -Wl,-dynamic-linker,/lib/ld-linux-riscv64xthead-lp64d.so.1 \
  --sysroot="$task_sdk/xuantie-gnu-toolchain/sysroot" --gcc-toolchain="$task_sdk/xuantie-gnu-toolchain" \
  -I"$task_sdk/test/leinao_test" -I"$task_repo/include" "$task_repo/examples/lsm_qemu/runner.cpp" \
  -lm -o "$task_artifacts/bin/rvne_runner" 2>&1 | tee "$task_artifacts/logs/runner_build.log"
cmake -S "$task_repo" -B "$task_build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE="-O2 -g0 -DNDEBUG" -DUSE_CUDA=OFF -DUSE_BACKTRACE=OFF \
  -DBUILD_NNET=OFF -DBUILD_TEST=ON -DPython_EXECUTABLE="$CONDA_PREFIX/bin/python" \
  2>&1 | tee "$task_artifacts/logs/configure.log"
cmake --build "$task_build" --target backend test_graph test_conv -j "${MOTION_BUILD_JOBS:-4}" \
  2>&1 | tee "$task_artifacts/logs/build.log"
export PYTHONPATH="$task_build:$task_repo/pyinfinitensor/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$task_build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export INFINI_RVNE_QEMU="$task_sdk/qemu/bin/qemu-riscv64"
export INFINI_RVNE_SYSROOT="$task_sdk/xuantie-gnu-toolchain/sysroot"
export INFINI_RVNE_RUNNER="$task_artifacts/bin/rvne_runner"
export INFINI_RVNE_TIMEOUT_MS=180000
python "$task_repo/examples/motion_qemu/export_model.py" --materials "$task_materials" \
  --output-dir "$task_artifacts/model" --chunk "${MOTION_CHUNK:-64}" \
  2>&1 | tee "$task_artifacts/logs/export.log"
task_sample_args=()
if [[ -n "${MOTION_SAMPLES:-}" ]]; then task_sample_args=(--samples "$MOTION_SAMPLES"); fi
python "$task_repo/examples/motion_qemu/validate.py" --model-dir "$task_artifacts/model" \
  --output-dir "$task_artifacts/validation" --chunk "${MOTION_CHUNK:-64}" "${task_sample_args[@]}" \
  2>&1 | tee "$task_artifacts/logs/validation.log"
ctest --test-dir "$task_build" --output-on-failure -R '^(test_graph|test_conv)$' \
  2>&1 | tee "$task_artifacts/logs/regression.log"
git -C "$task_repo" diff --check
