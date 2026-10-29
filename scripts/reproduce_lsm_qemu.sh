#!/usr/bin/env bash
# Build and validate the host InfiniTensor / QEMU RVNE prototype.
set -eo pipefail
task_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
task_parent=$(dirname "$task_repo")
task_artifacts=${LSM_QEMU_ARTIFACTS:-$task_parent/validation/lsm_mlp_qemu_runner}
task_sdk=${RVNE_SDK:-$task_parent/validation/rvne_m0_20261012/sdk/rvne/toolchain}
task_materials=${LSM_QEMU_MATERIALS:-$task_parent/20261009/analysis_work/materials/BSSL_LSM_bushu_light_quant_params_v5}
source /home/zhushuang/workspace/envs/set_a100_env.sh
conda activate infiniops_test
set -u
mkdir -p "$task_artifacts/logs" "$task_artifacts/bin" "$task_artifacts/tmp"
export TMPDIR="$task_artifacts/tmp"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
ulimit -c 0

"$task_sdk/llvm/bin/clang++" --target=riscv64-unknown-linux-gnu \
    -std=c++17 -march=rv64imafcvzne -O2 -ffp-contract=off -fuse-ld=lld \
    -Wl,-dynamic-linker,/lib/ld-linux-riscv64xthead-lp64d.so.1 \
    --sysroot="$task_sdk/xuantie-gnu-toolchain/sysroot" \
    --gcc-toolchain="$task_sdk/xuantie-gnu-toolchain" \
    -I"$task_sdk/test/leinao_test" -I"$task_repo/include" \
    "$task_repo/examples/lsm_qemu/runner.cpp" -o "$task_artifacts/bin/rvne_runner" \
    2>&1 | tee "$task_artifacts/logs/runner_build.log"
"$task_sdk/llvm/bin/llvm-objdump" -d "$task_artifacts/bin/rvne_runner" \
    > "$task_artifacts/logs/runner_disassembly.log"
"$task_sdk/llvm/bin/clang++" --target=riscv64-unknown-linux-gnu \
    -std=c++17 -march=rv64imafcvzne -O2 -ffp-contract=off -S \
    --sysroot="$task_sdk/xuantie-gnu-toolchain/sysroot" \
    --gcc-toolchain="$task_sdk/xuantie-gnu-toolchain" \
    -I"$task_sdk/test/leinao_test" -I"$task_repo/include" \
    "$task_repo/examples/lsm_qemu/runner.cpp" -o "$task_artifacts/logs/runner.s"

cmake -S "$task_repo" -B "$task_artifacts/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O2 -g0 -DNDEBUG" \
    -DUSE_CUDA=OFF -DUSE_BACKTRACE=OFF -DBUILD_NNET=OFF -DBUILD_TEST=ON \
    -DPython_EXECUTABLE="$CONDA_PREFIX/bin/python" \
    2>&1 | tee "$task_artifacts/logs/configure.log"
cmake --build "$task_artifacts/build" --target backend test_graph test_conv \
    -j "${LSM_QEMU_BUILD_JOBS:-4}" 2>&1 | tee "$task_artifacts/logs/build.log"
export PYTHONPATH="$task_artifacts/build:$task_repo/pyinfinitensor/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$task_artifacts/build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export INFINI_RVNE_QEMU="$task_sdk/qemu/bin/qemu-riscv64"
export INFINI_RVNE_SYSROOT="$task_sdk/xuantie-gnu-toolchain/sysroot"
export INFINI_RVNE_RUNNER="$task_artifacts/bin/rvne_runner"
export INFINI_RVNE_AUDIT="$task_artifacts/logs/qemu_launches.jsonl"
export INFINI_RVNE_TIMEOUT_MS=120000
: > "$INFINI_RVNE_AUDIT"

ctest --test-dir "$task_artifacts/build" --output-on-failure \
    -R '^(test_graph|test_conv)$' 2>&1 | tee "$task_artifacts/logs/regression.log"
python "$task_repo/pyinfinitensor/tests/test_api.py" \
    2>&1 | tee "$task_artifacts/logs/api_tests.log"
python "$task_repo/examples/lsm_qemu/export_model.py" --materials "$task_materials" \
    --output-dir "$task_artifacts/model" --count-mode "${LSM_COUNT_MODE:-wide}" \
    2>&1 | tee "$task_artifacts/logs/export.log"
python "$task_repo/examples/lsm_qemu/validate.py" --materials "$task_materials" --model "$task_artifacts/model/lsm_mlp_qemu.onnx" \
    --dataset "$task_artifacts/model/dataset.npz" --output-dir "$task_artifacts/validation" \
    --samples "${LSM_QEMU_SAMPLES:-25}" --policy "${LSM_STATE_POLICY:-independent}" \
    2>&1 | tee "$task_artifacts/logs/validation.log"
python "$task_repo/examples/lsm_qemu/infer.py" --model "$task_artifacts/model/lsm_mlp_qemu.onnx" \
    --dataset "$task_artifacts/model/dataset.npz" --sample 0 \
    --output-dir "$task_artifacts/standalone" 2>&1 | tee "$task_artifacts/logs/standalone.log"
git -C "$task_repo" diff --check
printf 'PASS: ONNX / InfiniTensor / QEMU / RVNE reproduction complete: %s\n' "$task_artifacts"
