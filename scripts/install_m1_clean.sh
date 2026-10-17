#!/usr/bin/env bash
# Build M1 from a source snapshot into a new prefix. No existing InfiniTensor,
# InfiniOps, or build directory is reused.
set -euo pipefail

usage() {
  echo "usage: $0 --source-dir DIR --work-dir DIR --python PYTHON --device nvidia|cambricon|ascend --model-dir DIR [--device-id N] [--build-only]" >&2
  exit 2
}

SOURCE= WORK= PYTHON= DEVICE= MODEL= DEVICE_ID=0 BUILD_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-dir) SOURCE=$2; shift 2;; --work-dir) WORK=$2; shift 2;;
    --python) PYTHON=$2; shift 2;; --device) DEVICE=$2; shift 2;;
    --model-dir) MODEL=$2; shift 2;; --device-id) DEVICE_ID=$2; shift 2;;
    --build-only) BUILD_ONLY=1; shift;; *) usage;;
  esac
done
[[ -n "$SOURCE" && -n "$WORK" && -n "$PYTHON" && -n "$DEVICE" && -n "$MODEL" ]] || usage
[[ ! -e "$WORK" ]] || { echo "work directory already exists: $WORK" >&2; exit 2; }
SOURCE=$(cd "$SOURCE" && pwd); MODEL=$(cd "$MODEL" && pwd)
mkdir -p "$WORK"
ROOT=$(cd "$WORK" && pwd)

# A source-only copy keeps the install validation independent of any current
# binary/module cache while retaining uncommitted M1 sources and the lock file.
mkdir "$ROOT/source"
tar -C "$SOURCE" --exclude=.git --exclude=build --exclude=build-* --exclude='*.so' \
  --exclude='__pycache__' --exclude=3rd-party -cf - . | tar -C "$ROOT/source" -xf -
mkdir -p "$ROOT/source/3rd-party"
cd "$ROOT"
LOCK="$ROOT/source/dependencies/m1-lock.json"
python3 - "$LOCK" <<'PY'
import hashlib,json,pathlib,sys
lock=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8-sig'))
patch=pathlib.Path(sys.argv[1]).parent/lock['InfiniOps_patch']
assert hashlib.sha256(patch.read_bytes()).hexdigest()==lock['InfiniOps_patch_sha256']
PY
clone_pin() {
  local repo=$1 sha=$2 dest=$3 local_source=${4:-}
  local bundle="$SOURCE/../git-bundles/$(basename "$dest").bundle"
  if [[ -f "$bundle" ]]; then
    git clone -q "$bundle" "$dest"
  elif [[ -n "$local_source" ]] && git -C "$local_source" cat-file -e "$sha^{commit}" 2>/dev/null; then
    git clone -q --no-hardlinks "$local_source" "$dest"
  else
    git clone -q "https://github.com/$repo.git" "$dest"
  fi
  git -C "$dest" checkout -q "$sha"
  test "$(git -C "$dest" rev-parse HEAD)" = "$sha"
}
RT_SHA=$(python3 -c 'import json; print(json.load(open("'"$LOCK"'"))["InfiniRT"])')
OPS_SHA=$(python3 -c 'import json; print(json.load(open("'"$LOCK"'"))["InfiniOps_base"])')
clone_pin InfiniTensor/InfiniRT "$RT_SHA" InfiniRT "$SOURCE/../InfiniRT"
clone_pin InfiniTensor/InfiniOps "$OPS_SHA" InfiniOps "$SOURCE/../InfiniOps"
git -C InfiniOps apply "$ROOT/source/dependencies/InfiniOps-m1.patch"
# The source snapshot has no .git directory.  Fresh-clone the three CMake
# dependencies at immutable revisions, preferring the verified local object
# stores and using GitHub only when a pinned object is unavailable locally.
clone_pin pybind/pybind11 0bd8896a4010f2d91b2340570c24fa08606ec406 "$ROOT/source/3rd-party/pybind11" "$SOURCE/3rd-party/pybind11"
clone_pin ArthurSonzogni/nlohmann_json_cmake_fetchcontent 13132dd361c8c5b5753983d5186cf54f689d90f9 "$ROOT/source/3rd-party/nlohmann_json_cmake_fetchcontent" "$SOURCE/3rd-party/nlohmann_json_cmake_fetchcontent"
clone_pin google/googletest b796f7d44681514f58a683a3a71ff17c94edb0c1 "$ROOT/source/3rd-party/googletest" "$SOURCE/3rd-party/googletest"
ABI=$($PYTHON -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
PREFIX="$ROOT/prefix"; mkdir "$PREFIX"
export PATH="$(dirname "$PYTHON"):$PATH"
export LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}"
python3 - "$DEVICE" "$ROOT/selected_ops.json" <<'PY'
import json,sys
names='add mul relu gelu cat matmul softmax gather onnx_gather cast index_select native_layer_norm sub div erf sqrt pow where eq lt gt expand_copy permute_copy slice_copy native_batch_norm max_pool2d_with_indices avg_pool2d'.split()
selection = {name: {'implementations': [8]} for name in names}
# master graph/copy regression fixtures use the native RMSNorm implementation.
selection['rms_norm'] = {'implementations': [0]}
json.dump(selection,open(sys.argv[2],'w'),indent=2)
PY
UPPER=$(tr '[:lower:]' '[:upper:]' <<<"$DEVICE")
EXTRA=()
if [[ "$DEVICE" == nvidia ]]; then
  CUDA_HOME=${CUDA_HOME:?CUDA_HOME is required for NVIDIA clean install}
  CUTLASS_COMMIT=$(python3 -c 'import json; print(json.load(open("'"$LOCK"'"))["CUTLASS"])')
  CUTLASS_SHA=$(python3 -c 'import json; print(json.load(open("'"$LOCK"'"))["CUTLASS_archive_sha256"])')
  CUTLASS_ARCHIVE="$SOURCE/../cutlass-$CUTLASS_COMMIT.tar.gz"
  echo "$CUTLASS_SHA  $CUTLASS_ARCHIVE" | sha256sum -c -
  mkdir -p "$ROOT/offline"
  tar -xzf "$CUTLASS_ARCHIVE" -C "$ROOT/offline"
  CUTLASS_SOURCE="$ROOT/offline/cutlass-src"
  [[ -d "$CUTLASS_SOURCE/include/cutlass" ]] || {
    echo "CUTLASS source is missing: $CUTLASS_SOURCE" >&2
    exit 2
  }
  EXTRA=(-DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:?CUDA_ARCH is required for NVIDIA clean install}"
         -DCUDAToolkit_ROOT="$CUDA_HOME" -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc"
         -DFETCHCONTENT_SOURCE_DIR_CUTLASS="$CUTLASS_SOURCE")
fi
cmake -S InfiniRT -B build-rt -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=$ABI" -DWITH_"$UPPER"=ON -DINFINI_RT_BUILD_TESTING=ON -DPython_EXECUTABLE="$PYTHON" "${EXTRA[@]}"
cmake --build build-rt -j4; cmake --install build-rt
cmake -S InfiniOps -B build-ops -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" -DINFINI_RT_ROOT="$PREFIX" -DWITH_"$UPPER"=ON -DWITH_TORCH=ON -DGENERATE_PYTHON_BINDINGS=OFF -DBUILD_ASCEND_CUSTOM=OFF -DINFINI_OPS_OPS="$ROOT/selected_ops.json" -DPython_EXECUTABLE="$PYTHON" "${EXTRA[@]}"
cmake --build build-ops -j4; cmake --install build-ops
cmake -S "$ROOT/source" -B build-it -DCMAKE_BUILD_TYPE=Release -DUSE_BACKTRACE=OFF -DBUILD_TEST=ON -DUSE_INFINIOPS_KERNELS=ON -DUSE_INFINIOPS_ATEN_KERNELS=ON -DINFINIOPS_ROOT="$PREFIX" -DINFINIRT_ROOT="$PREFIX" -DPython_EXECUTABLE="$PYTHON" "${EXTRA[@]}"
cmake --build build-it -j4
cp build-it/backend*.so build-it/libInfiniTensor.so "$ROOT/source/pyinfinitensor/src/pyinfinitensor/"
export PYTHONPATH="$ROOT/source/pyinfinitensor/src"
case "$DEVICE" in
  cambricon) export INFINIOPS_PROVIDER_MODULES=torch_mlu ;;
  ascend) export INFINIOPS_PROVIDER_MODULES=torch_npu ;;
esac
if [[ "$BUILD_ONLY" == 1 ]]; then
  echo "M1 clean build completed; verification pending: $ROOT"
  exit 0
fi
$PYTHON "$ROOT/source/scripts/check_install.py" --backend "$DEVICE" --device "$DEVICE_ID" --require-infini --model-dir "$MODEL" --json > "$ROOT/install-check.json"
$PYTHON "$ROOT/source/scripts/accept_m1.py" --stage install --model-dir "$MODEL" \
  --device "$DEVICE" --device-id "$DEVICE_ID" --install-prefix "$ROOT" --output "$ROOT/report"
ctest --test-dir build-it --output-on-failure > "$ROOT/ctest.txt"
ctest --test-dir build-rt --output-on-failure > "$ROOT/infinirt-ctest.txt"
echo "M1 clean installation passed: $ROOT"
