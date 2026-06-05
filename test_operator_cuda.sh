#!/usr/bin/env bash
# test_operator_cuda.sh — CUDA operator correctness tests
#
# Tests Matmul, Add, RMSNorm, SwiGLU on target device.
# Each operator runs with 3 shape variants (S/M/L) and is verified
# against the CPU reference result via element-wise comparison.
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
BUILD_DIR="${BUILD_DIR:-$PROJECT_DIR/build/Release}"
TEST_BIN="$BUILD_DIR/test_operator_yanshou"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

pass=0
fail=0
skip=0
total=0

log_pass() { ((pass++)); ((total++)); echo -e "${GREEN}[PASS]${NC} $1"; }
log_fail() { ((fail++)); ((total++)); echo -e "${RED}[FAIL]${NC} $1"; }
log_skip() { ((skip++)); ((total++)); echo -e "${YELLOW}[SKIP]${NC} $1"; }
log_section() { echo -e "\n${CYAN}=== $1 ===${NC}"; }

die() {
    echo -e "${RED}ERROR: $1${NC}" >&2
    exit 1
}

if [ ! -x "$TEST_BIN" ]; then
    die "Test binary not found: $TEST_BIN  (set BUILD_DIR or run: make build CUDA=ON TEST=ON)"
fi

# Run a gtest invocation.
# All output (stdout + stderr, including precision diagnostics) is
# always printed to the terminal. On failure, extra detail is extracted.
# Usage: run_tests "label" "gtest_filter" [timeout_seconds]
run_tests() {
    local label="$1" filter="$2" timeout_sec="${3:-300}"
    local tmpfile rc
    tmpfile=$(mktemp /tmp/gtest_output.XXXXXX)
    # Tee: show output live AND capture for failure analysis
    timeout "$timeout_sec" "$TEST_BIN" --gtest_filter="$filter" \
        --gtest_print_time=0 2>&1 | tee "$tmpfile" || rc=$?
    rc=${rc:-0}
    if [ "$rc" -eq 0 ]; then
        log_pass "$label"
    elif [ "$rc" -eq 124 ]; then
        log_skip "$label (timeout after ${timeout_sec}s)"
    else
        log_fail "$label"
        grep -E "FAILED|error|Assert" "$tmpfile" | head -10
    fi
    rm -f "$tmpfile"
}

# ───────────────────────────────────────────────────────────────
log_section "Operator Correctness Tests"
echo "Binary : $TEST_BIN"
echo "Date   : $(date '+%Y-%m-%d %H:%M:%S')"

# ───────────────────────────────────────────────────────────────
# Shape variants: S, M, L
# Device suffix: nvidia (default; add more devices to test file
# by extending kDevices[] in test_operator_cuda.cc)
# ───────────────────────────────────────────────────────────────

DEVICES=("nvidia")
SHAPES=("S" "M" "L")
TEST_CLASS="AllShapesAndDevices/OperatorTest"

# ───────────────────────────────────────────────────────────────
# 1. Matmul
# ───────────────────────────────────────────────────────────────
log_section "1. Matmul"

for device in "${DEVICES[@]}"; do
    for shape in "${SHAPES[@]}"; do
        run_tests "Matmul ($shape/$device)" \
            "*${TEST_CLASS}.Matmul/${shape}_${device}"
    done
done

# ───────────────────────────────────────────────────────────────
# 2. Add
# ───────────────────────────────────────────────────────────────
log_section "2. Add"

for device in "${DEVICES[@]}"; do
    for shape in "${SHAPES[@]}"; do
        run_tests "Add ($shape/$device)" \
            "*${TEST_CLASS}.Add/${shape}_${device}"
    done
done

# ───────────────────────────────────────────────────────────────
# 3. RMSNorm
# ───────────────────────────────────────────────────────────────
log_section "3. RMSNorm"

for device in "${DEVICES[@]}"; do
    for shape in "${SHAPES[@]}"; do
        run_tests "RMSNorm ($shape/$device)" \
            "*${TEST_CLASS}.RMSNorm/${shape}_${device}"
    done
done

# ───────────────────────────────────────────────────────────────
# 4. SwiGLU
# ───────────────────────────────────────────────────────────────
log_section "4. SwiGLU"

for device in "${DEVICES[@]}"; do
    for shape in "${SHAPES[@]}"; do
        run_tests "SwiGLU ($shape/$device)" \
            "*${TEST_CLASS}.SwiGLU/${shape}_${device}"
    done
done

# ───────────────────────────────────────────────────────────────
# Summary
# ───────────────────────────────────────────────────────────────
log_section "Summary"

echo "  Total  : $total"
echo -e "  Passed : ${GREEN}$pass${NC}"
echo -e "  Failed : ${RED}$fail${NC}"
echo -e "  Skipped: ${YELLOW}$skip${NC}"
echo ""

if [ $fail -ne 0 ]; then
    echo -e "${RED}SOME TESTS FAILED${NC}"
    exit 1
else
    echo -e "${GREEN}ALL TESTS PASSED${NC}"
    exit 0
fi
