#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:-$(dirname "$ROOT")/.venv}"
cd "$ROOT"
source examples/dynamic_shape/env_cuda.sh
source "$VENV/bin/activate"
nvcc --version
cmake -S . -B build/CUDA \
    -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DBUILD_TEST=ON \
    -DUSE_BACKTRACE=ON -DCUDAToolkit_ROOT="$CUDA_HOME" \
    -DINFINITENSOR_CUDA_ARCHITECTURES="${CUDA_ARCHS:-75}"
cmake --build build/CUDA -j "${JOBS:-2}"
cp build/CUDA/backend*.so pyinfinitensor/src/pyinfinitensor/
python -m pip install --no-deps -e pyinfinitensor
python -m pytest -q pyinfinitensor/tests/test_dynamic_shape_cuda.py
python examples/dynamic_shape/cuda_demo.py
