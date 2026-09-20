#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:-$(dirname "$ROOT")/.venv}"
cd "$ROOT"
source examples/dynamic_shape/env_cuda.sh
source "$VENV/bin/activate"
mkdir -p artifacts/dynamic_shape
python -c 'from pyinfinitensor import backend; assert hasattr(backend, "CudaRuntime"), "CUDA build required"'
python -m pytest -q pyinfinitensor/tests/test_dynamic_shape.py \
    pyinfinitensor/tests/test_dynamic_shape_cuda.py \
    pyinfinitensor/tests/test_onnxstub.py pyinfinitensor/tests/test_onnx.py \
    pyinfinitensor/tests/test_api.py 2>&1 | tee artifacts/dynamic_shape/cuda-full-python-tests.log
ctest --test-dir build/CUDA --output-on-failure --timeout 120 \
    2>&1 | tee artifacts/dynamic_shape/cuda-cpp-tests.log
# CUDA 12.8's host backtrace collection retains Python frames (and their GPU
# owners). Disable backtrace collection, not memory instrumentation or leaks.
compute-sanitizer --show-backtrace no --tool memcheck --leak-check full \
    --error-exitcode 99 python examples/dynamic_shape/cuda_demo.py \
    --output artifacts/dynamic_shape/cuda-sanitized-results.json \
    2>&1 | tee artifacts/dynamic_shape/cuda-memcheck.log
compute-sanitizer --show-backtrace no --tool memcheck --leak-check full \
    --error-exitcode 99 python examples/dynamic_shape/cuda_demo.py --naive \
    --output artifacts/dynamic_shape/cuda-naive-sanitized-results.json \
    2>&1 | tee artifacts/dynamic_shape/cuda-naive-memcheck.log
