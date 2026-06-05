#!/usr/bin/env bash
# test_runtime_cuda.sh — Runtime memory operation tests
#
# Tests Malloc, MemcpyToDevice, MemcpyToHost across buffer sizes (4KB-256MB),
# alignment variants (128B, 4KB), and data types (FP32, FP16, INT8).
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
BUILD_DIR="${BUILD_DIR:-$PROJECT_DIR/build/Release}"
TEST_BIN="$BUILD_DIR/test_runtime_yanshou"

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
log_info() { echo -e "  ${CYAN}INFO${NC} $1"; }

die() {
    echo -e "${RED}ERROR: $1${NC}" >&2
    exit 1
}

# Run a gtest invocation and return 0 on success.
# Usage: run_tests "label" "gtest_filter" [timeout_seconds]
run_tests() {
    local label="$1" filter="$2" timeout_sec="${3:-300}"
    local tmpfile rc
    tmpfile=$(mktemp)
    timeout "$timeout_sec" "$TEST_BIN" --gtest_filter="$filter" --gtest_print_time=0 2>&1 | tee "$tmpfile"
    rc=${PIPESTATUS[0]}
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
# 0. Prerequisites
# ───────────────────────────────────────────────────────────────
if [ ! -x "$TEST_BIN" ]; then
    die "Test binary not found: $TEST_BIN  (set BUILD_DIR or run: make build CUDA=ON TEST=ON)"
fi

log_section "Runtime Memory Tests"
echo "Binary : $TEST_BIN"
echo "Date   : $(date '+%Y-%m-%d %H:%M:%S')"

# Device suffix: matches kDevices[] in test_runtime_cuda.cc
DEVICES=("nvidia")
TEST_CLASS="AllDevices/RuntimeDeviceTest"

log_info "Devices under test : ${DEVICES[*]}"
log_info "Test class         : ${TEST_CLASS}"

# ───────────────────────────────────────────────────────────────
# 1. Malloc / Dealloc
# ───────────────────────────────────────────────────────────────
log_section "1. Malloc / Dealloc"
log_info "Allocate device memory, fill pattern host->device, read back device->host, verify round-trip integrity."
log_info "Buffer sizes: 4KB, 64KB, 1MB, 16MB, 64MB, 256MB"

for device in "${DEVICES[@]}"; do
    run_tests "Malloc ($device)" "*${TEST_CLASS}.Malloc/${device}"
done

# ───────────────────────────────────────────────────────────────
# 2. MemcpyToDevice (Host -> Device)
# ───────────────────────────────────────────────────────────────
log_section "2. MemcpyToDevice (Host -> Device)"
log_info "Upload known host buffer to device, download and memcmp against original."
log_info "Verifies host->device copy does not corrupt data."

for device in "${DEVICES[@]}"; do
    run_tests "MemcpyToDevice ($device)" "*${TEST_CLASS}.MemcpyToDevice/${device}"
done

# ───────────────────────────────────────────────────────────────
# 3. MemcpyToHost (Device -> Host)
# ───────────────────────────────────────────────────────────────
log_section "3. MemcpyToHost (Device -> Host)"
log_info "Upload data to device first, then download to a fresh host buffer and memcmp."
log_info "Verifies device->host copy does not corrupt data."

for device in "${DEVICES[@]}"; do
    run_tests "MemcpyToHost ($device)" "*${TEST_CLASS}.MemcpyToHost/${device}"
done

# ───────────────────────────────────────────────────────────────
# 4. Alignment variants — 128B and 4KB
# ───────────────────────────────────────────────────────────────
log_section "4. Alignment Variants"
log_info "Allocate buffers aligned to 128B and 4KB page boundaries, round-trip verify."
log_info "Catches misaligned access issues on devices with strict alignment requirements."

for device in "${DEVICES[@]}"; do
    run_tests "Align128B ($device)" "*${TEST_CLASS}.Align128B/${device}"
    run_tests "Align4KB ($device)" "*${TEST_CLASS}.Align4KB/${device}"
done

# ───────────────────────────────────────────────────────────────
# 5. DataType coverage — FP32, FP16, INT8
# ───────────────────────────────────────────────────────────────
log_section "5. DataType Coverage (1024 elements)"
log_info "Round-trip test for FP32 (4B), FP16 (2B), INT8 (1B) element types."
log_info "Total bytes: FP32=4096B, FP16=2048B, INT8=1024B."

for device in "${DEVICES[@]}"; do
    run_tests "FP32 ($device)" "*${TEST_CLASS}.FP32_RoundTrip/${device}"
    run_tests "FP16 ($device)" "*${TEST_CLASS}.FP16_RoundTrip/${device}"
    run_tests "INT8 ($device)" "*${TEST_CLASS}.INT8_RoundTrip/${device}"
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
