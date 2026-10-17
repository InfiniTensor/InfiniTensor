#!/usr/bin/env bash
# Verify a build produced by install_m1_clean.sh --build-only from outside the
# source tree.  Provider SDK paths may be supplied through the two M1_* extras.
set -euo pipefail

usage() {
  echo "usage: $0 --work-dir DIR --python PYTHON --device nvidia|cambricon|ascend --device-id N --model-dir DIR --result-dir DIR" >&2
  exit 2
}

WORK= PYTHON= DEVICE= MODEL= RESULT= DEVICE_ID=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-dir) WORK=$2; shift 2;; --python) PYTHON=$2; shift 2;;
    --device) DEVICE=$2; shift 2;; --device-id) DEVICE_ID=$2; shift 2;;
    --model-dir) MODEL=$2; shift 2;; --result-dir) RESULT=$2; shift 2;;
    *) usage;;
  esac
done
[[ -n "$WORK" && -n "$PYTHON" && -n "$DEVICE" && -n "$MODEL" && -n "$RESULT" ]] || usage
ROOT=$(cd "$WORK" && pwd)
MODEL=$(cd "$MODEL" && pwd)
mkdir -p "$RESULT"
RESULT=$(cd "$RESULT" && pwd)
PREFIX="$ROOT/prefix"
SOURCE="$ROOT/source"
compgen -G "$SOURCE/pyinfinitensor/src/pyinfinitensor/backend*.so" >/dev/null || {
  echo "clean backend is missing under $SOURCE" >&2
  exit 2
}

export PYTHONPATH="$SOURCE/pyinfinitensor/src${M1_DEPENDENCY_PYTHONPATH:+:$M1_DEPENDENCY_PYTHONPATH}"
export LD_LIBRARY_PATH="$PREFIX/lib${M1_VENDOR_LD_LIBRARY_PATH:+:$M1_VENDOR_LD_LIBRARY_PATH}"
case "$DEVICE" in
  cambricon) export INFINIOPS_PROVIDER_MODULES=torch_mlu ;;
  ascend) export INFINIOPS_PROVIDER_MODULES=torch_npu ;;
  nvidia) unset INFINIOPS_PROVIDER_MODULES || true ;;
  *) usage ;;
esac

cd "$ROOT"
"$PYTHON" "$SOURCE/scripts/check_install.py" --backend "$DEVICE" --device "$DEVICE_ID" \
  --require-infini --model-dir "$MODEL" --json > "$RESULT/install-check.json"
"$PYTHON" "$SOURCE/scripts/accept_m1.py" --stage install --model-dir "$MODEL" \
  --device "$DEVICE" --device-id "$DEVICE_ID" --install-prefix "$ROOT" --output "$RESULT"
export INFINITENSOR_REQUIRE_GRAPH_CAPTURE=1
ctest --test-dir "$ROOT/build-it" --output-on-failure > "$RESULT/ctest.txt"
ctest --test-dir "$ROOT/build-rt" --output-on-failure > "$RESULT/infinirt-ctest.txt"
"$PYTHON" -m unittest discover -s "$SOURCE/scripts" -p test_accept_m1.py \
  > "$RESULT/test_accept_m1.log" 2>&1

"$PYTHON" "$SOURCE/scripts/validate_accelerator_ops.py" --device "$DEVICE" \
  --output "$RESULT/accelerator-ops.json" > "$RESULT/accelerator-ops.log" 2>&1
if grep -Eiq 'fall.?back.{0,100}(run on the CPU|to (the )?CPU)|CPU.{0,30}fall.?back' "$RESULT/accelerator-ops.log"; then
  echo "unexpected CPU fallback in accelerator operator verification" >&2
  exit 1
fi
# The certified drivers expose capture on all three backends. Never let a
# regression turn a required capture check into a capability skip.
export INFINITENSOR_REQUIRE_GRAPH_CAPTURE=1
for pattern in test_copy_graph_capture.py test_workspace_graph_capture.py; do
  "$PYTHON" -m unittest discover -s "$SOURCE/pyinfinitensor/tests" -p "$pattern" \
    > "$RESULT/${pattern%.py}.log" 2>&1
done
unset INFINITENSOR_REQUIRE_GRAPH_CAPTURE

# Exercise the public offline commands from a directory outside the source.
OUTSIDE="$ROOT/outside-client"
mkdir -p "$OUTSIDE"
cat > "$OUTSIDE/documents.jsonl" <<'EOF'
{"id":"doc-2","text":"机器学习系统需要稳定的推理运行时。"}
{"id":"doc-1","text":"中文向量可用于语义检索。"}
EOF
(
  cd "$OUTSIDE"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PYTHON" "$SOURCE/examples/python/encode_jsonl.py" \
    --model-dir "$MODEL" --input documents.jsonl --output vectors.npz \
    --device "$DEVICE" --device-id "$DEVICE_ID"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PYTHON" "$SOURCE/examples/python/query_jsonl.py" \
    --model-dir "$MODEL" --vectors vectors.npz --query "如何进行中文语义检索？" --top-k 2 \
    --device "$DEVICE" --device-id "$DEVICE_ID" > query.json
)
test -s "$OUTSIDE/vectors.npz" -a -s "$OUTSIDE/query.json"

"$PYTHON" - "$ROOT" "$RESULT/clean-provenance.json" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
paths = list((root/'source/pyinfinitensor/src/pyinfinitensor').glob('backend*.so')) + [
         root/'source/pyinfinitensor/src/pyinfinitensor/libInfiniTensor.so',
         root/'prefix/lib/libinfinirt.so', root/'prefix/lib/libinfiniops.so']
data = {'status': 'PASS', 'work_dir': str(root),
        'artifacts': {str(p.relative_to(root)): sha(p) for p in paths if p.is_file()}}
pathlib.Path(sys.argv[2]).write_text(json.dumps(data, indent=2, sort_keys=True), encoding='utf-8')
PY
echo "M1 clean verification passed: $ROOT"
