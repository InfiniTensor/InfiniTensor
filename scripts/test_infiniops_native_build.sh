#!/usr/bin/env bash
# Run from a checkout with its submodules initialized. No Torch installation
# or previously generated provider headers are needed.
set -euo pipefail
src=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/infinitensor-native.XXXXXX")
py=${PYTHON_EXECUTABLE:-python3}
jobs=${BUILD_JOBS:-2}
echo "Clean native build: $work"

mkdir -p "$work/ops" "$work/rt"
# InfiniOps code generation writes into its source tree. Archive the pinned
# commits so an earlier WITH_TORCH build cannot contaminate this regression.
git -C "$src/3rd-party/InfiniOps" archive HEAD | tar -x -C "$work/ops"
git -C "$src/3rd-party/InfiniRT" archive HEAD | tar -x -C "$work/rt"
cmake -S "$work/rt" -B "$work/build-rt" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$work/prefix" \
    -DWITH_CPU=ON -DPython_EXECUTABLE="$py"
cmake --build "$work/build-rt" -j "$jobs"
cmake --install "$work/build-rt"
cmake -S "$work/ops" -B "$work/build-ops" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$work/prefix" \
    -DWITH_CPU=ON -DWITH_TORCH=OFF -DGENERATE_PYTHON_BINDINGS=OFF \
    -DINFINI_RT_ROOT="$work/prefix" -DPython_EXECUTABLE="$py"
cmake --build "$work/build-ops" -j "$jobs"
cmake --install "$work/build-ops"
test ! -d "$work/ops/generated/base"

cmake -S "$src" -B "$work/build-it" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=ON -DUSE_BACKTRACE=OFF \
    -DUSE_INFINIOPS_KERNELS=ON -DUSE_INFINIOPS_ATEN_KERNELS=OFF \
    -DINFINIOPS_ROOT="$work/prefix" -DINFINIRT_ROOT="$work/prefix" \
    -DINFINIOPS_CXX11_ABI=1 -DPython_EXECUTABLE="$py"
cmake --build "$work/build-it" -j "$jobs"
"$work/build-it/test_copy_graph_capture" \
    --gtest_filter=InfiniCopyLifetimeTest.*
