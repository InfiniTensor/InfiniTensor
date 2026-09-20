#!/usr/bin/env bash
set -euo pipefail
# Run from an existing checkout; use HTTPS overrides for public SSH submodules.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:-$(dirname "$ROOT")/.venv}"
cd "$ROOT"
git -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --recursive
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install -r examples/dynamic_shape/requirements.txt
cmake -S . -B build/Release -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=ON -DUSE_BACKTRACE=ON
cmake --build build/Release -j "${JOBS:-2}"
cp build/Release/backend*.so pyinfinitensor/src/pyinfinitensor/
python -m pip install -e pyinfinitensor
make test-dynamic-shape
