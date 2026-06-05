#include "core/blob.h"
#include "core/runtime.h"
#include "device.h"
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace infini {
namespace test {

// ---------------------------------------------------------------------------
// Helpers: generate host buffers with a fixed seed
// ---------------------------------------------------------------------------

static void fillPattern(void *buf, size_t bytes, uint32_t seed) {
    auto *p = static_cast<uint8_t *>(buf);
    uint32_t state = seed;
    for (size_t i = 0; i < bytes; ++i) {
        state = state * 1103515245u + 12345u;
        p[i] = static_cast<uint8_t>(state >> 16);
    }
}

static bool verifyPattern(const void *buf, size_t bytes, uint32_t seed) {
    auto *p = static_cast<const uint8_t *>(buf);
    uint32_t state = seed;
    for (size_t i = 0; i < bytes; ++i) {
        state = state * 1103515245u + 12345u;
        if (p[i] != static_cast<uint8_t>(state >> 16))
            return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Device list — extend this to test additional devices
// ---------------------------------------------------------------------------

static const Device::Type kDevices[] = {
    Device::Type::kNvidia,
};

// ---------------------------------------------------------------------------
// Base fixture — parameterized by Device::Type
// ---------------------------------------------------------------------------

class RuntimeDeviceTest : public ::testing::TestWithParam<Device::Type> {
  protected:
    void SetUp() override {
        runtime = make_ref<RuntimeObj>(Device(GetParam()));
    }

    void TearDown() override { runtime.reset(); }

    Runtime runtime;
};

INSTANTIATE_TEST_SUITE_P(
    AllDevices, RuntimeDeviceTest, ::testing::ValuesIn(kDevices),
    [](const ::testing::TestParamInfo<Device::Type> &info) {
        return std::string(Device::StringFromType(info.param));
    });

// ---------------------------------------------------------------------------
// Size configurations
// ---------------------------------------------------------------------------

struct SizeConfig {
    const char *name;
    size_t bytes;
};

static const SizeConfig kSizes[] = {
    {"4KB", 4 * 1024},          {"64KB", 64 * 1024},
    {"1MB", 1024 * 1024},       {"16MB", 16 * 1024 * 1024},
    {"64MB", 64 * 1024 * 1024}, {"256MB", 256 * 1024 * 1024},
};

static constexpr uint32_t kSeed = 0xDEADBEEF;

// ============================================================
// 1. Malloc + Dealloc
// ============================================================

TEST_P(RuntimeDeviceTest, Malloc) {
    for (const auto &cfg : kSizes) {
        void *devPtr = runtime->alloc(cfg.bytes);
        ASSERT_NE(devPtr, nullptr) << cfg.name << ": alloc failed";

        std::vector<uint8_t> hostSrc(cfg.bytes);
        fillPattern(hostSrc.data(), cfg.bytes, kSeed);
        runtime->copyBlobFromCPU(devPtr, hostSrc.data(), cfg.bytes);

        std::vector<uint8_t> hostDst(cfg.bytes);
        runtime->copyBlobToCPU(hostDst.data(), devPtr, cfg.bytes);

        EXPECT_TRUE(verifyPattern(hostDst.data(), cfg.bytes, kSeed))
            << cfg.name << " (" << cfg.bytes << " bytes): round-trip mismatch";
        if (verifyPattern(hostDst.data(), cfg.bytes, kSeed))
            std::cout << "[  PASSED  ] " << cfg.name << " (" << cfg.bytes
                      << " bytes): data error = 0" << std::endl;

        runtime->dealloc(devPtr);
    }
}

// ============================================================
// 2. MemcpyToDevice — host->device
// ============================================================

TEST_P(RuntimeDeviceTest, MemcpyToDevice) {
    for (const auto &cfg : kSizes) {
        void *devPtr = runtime->alloc(cfg.bytes);
        ASSERT_NE(devPtr, nullptr) << cfg.name << ": alloc failed";

        std::vector<uint8_t> hostSrc(cfg.bytes);
        fillPattern(hostSrc.data(), cfg.bytes, kSeed);
        runtime->copyBlobFromCPU(devPtr, hostSrc.data(), cfg.bytes);

        std::vector<uint8_t> hostDst(cfg.bytes);
        runtime->copyBlobToCPU(hostDst.data(), devPtr, cfg.bytes);

        int cmp = std::memcmp(hostSrc.data(), hostDst.data(), cfg.bytes);
        EXPECT_EQ(cmp, 0)
            << cfg.name << " (" << cfg.bytes
            << " bytes): host->device->host mismatch";
        if (cmp == 0)
            std::cout << "[  PASSED  ] " << cfg.name << " (" << cfg.bytes
                      << " bytes): data error = 0" << std::endl;

        runtime->dealloc(devPtr);
    }
}

// ============================================================
// 3. MemcpyToHost — device->host
// ============================================================

TEST_P(RuntimeDeviceTest, MemcpyToHost) {
    for (const auto &cfg : kSizes) {
        void *devPtr = runtime->alloc(cfg.bytes);
        ASSERT_NE(devPtr, nullptr) << cfg.name << ": alloc failed";

        std::vector<uint8_t> original(cfg.bytes);
        fillPattern(original.data(), cfg.bytes, kSeed);
        runtime->copyBlobFromCPU(devPtr, original.data(), cfg.bytes);

        std::vector<uint8_t> downloaded(cfg.bytes);
        runtime->copyBlobToCPU(downloaded.data(), devPtr, cfg.bytes);

        int cmp = std::memcmp(original.data(), downloaded.data(), cfg.bytes);
        EXPECT_EQ(cmp, 0)
            << cfg.name << " (" << cfg.bytes
            << " bytes): device->host mismatch";
        if (cmp == 0)
            std::cout << "[  PASSED  ] " << cfg.name << " (" << cfg.bytes
                      << " bytes): data error = 0" << std::endl;

        runtime->dealloc(devPtr);
    }
}

// ============================================================
// 4. Alignment tests — 128B and 4KB aligned buffers
// ============================================================

TEST_P(RuntimeDeviceTest, Align128B) {
    for (const auto &cfg : kSizes) {
        const size_t align = 128;
        size_t bytes = (cfg.bytes + align - 1) & ~(align - 1);

        void *devPtr = runtime->alloc(bytes);
        ASSERT_NE(devPtr, nullptr) << cfg.name << ": alloc failed";

        std::vector<uint8_t> src(bytes);
        fillPattern(src.data(), bytes, kSeed + 1);
        runtime->copyBlobFromCPU(devPtr, src.data(), bytes);

        std::vector<uint8_t> dst(bytes);
        runtime->copyBlobToCPU(dst.data(), devPtr, bytes);

        int cmp = std::memcmp(src.data(), dst.data(), bytes);
        EXPECT_EQ(cmp, 0)
            << "128B-aligned " << cfg.name << " (" << bytes
            << " bytes) round-trip failed";
        if (cmp == 0)
            std::cout << "[  PASSED  ] 128B-aligned " << cfg.name << " ("
                      << bytes << " bytes): data error = 0" << std::endl;

        runtime->dealloc(devPtr);
    }
}

TEST_P(RuntimeDeviceTest, Align4KB) {
    for (const auto &cfg : kSizes) {
        const size_t align = 4096;
        size_t bytes = (cfg.bytes + align - 1) & ~(align - 1);

        void *devPtr = runtime->alloc(bytes);
        ASSERT_NE(devPtr, nullptr) << cfg.name << ": alloc failed";

        std::vector<uint8_t> src(bytes);
        fillPattern(src.data(), bytes, kSeed + 2);
        runtime->copyBlobFromCPU(devPtr, src.data(), bytes);

        std::vector<uint8_t> dst(bytes);
        runtime->copyBlobToCPU(dst.data(), devPtr, bytes);

        int cmp = std::memcmp(src.data(), dst.data(), bytes);
        EXPECT_EQ(cmp, 0)
            << "4KB-aligned " << cfg.name << " (" << bytes
            << " bytes) round-trip failed";
        if (cmp == 0)
            std::cout << "[  PASSED  ] 4KB-aligned " << cfg.name << " ("
                      << bytes << " bytes): data error = 0" << std::endl;

        runtime->dealloc(devPtr);
    }
}

// ============================================================
// 5. DataType tests — FP32, FP16, INT8 (1024 elements each)
// ============================================================

static constexpr size_t kDtypeElements = 1024;

TEST_P(RuntimeDeviceTest, FP32_RoundTrip) {
    size_t bytes = kDtypeElements * sizeof(float);
    void *devPtr = runtime->alloc(bytes);
    ASSERT_NE(devPtr, nullptr);

    std::vector<float> src(kDtypeElements);
    fillPattern(src.data(), bytes, kSeed);
    runtime->copyBlobFromCPU(devPtr, src.data(), bytes);

    std::vector<float> dst(kDtypeElements);
    runtime->copyBlobToCPU(dst.data(), devPtr, bytes);

    int cmp = std::memcmp(src.data(), dst.data(), bytes);
    EXPECT_EQ(cmp, 0) << "FP32 round-trip mismatch";
    if (cmp == 0)
        std::cout << "[  PASSED  ] FP32 data error = 0 (" << bytes << " bytes)"
                  << std::endl;
    runtime->dealloc(devPtr);
}

TEST_P(RuntimeDeviceTest, FP16_RoundTrip) {
    size_t bytes = kDtypeElements * sizeof(uint16_t);
    void *devPtr = runtime->alloc(bytes);
    ASSERT_NE(devPtr, nullptr);

    std::vector<uint16_t> src(kDtypeElements);
    fillPattern(src.data(), bytes, kSeed);
    runtime->copyBlobFromCPU(devPtr, src.data(), bytes);

    std::vector<uint16_t> dst(kDtypeElements);
    runtime->copyBlobToCPU(dst.data(), devPtr, bytes);

    int cmp = std::memcmp(src.data(), dst.data(), bytes);
    EXPECT_EQ(cmp, 0) << "FP16 round-trip mismatch";
    if (cmp == 0)
        std::cout << "[  PASSED  ] FP16 data error = 0 (" << bytes << " bytes)"
                  << std::endl;
    runtime->dealloc(devPtr);
}

TEST_P(RuntimeDeviceTest, INT8_RoundTrip) {
    size_t bytes = kDtypeElements * sizeof(int8_t);
    void *devPtr = runtime->alloc(bytes);
    ASSERT_NE(devPtr, nullptr);

    std::vector<int8_t> src(kDtypeElements);
    fillPattern(src.data(), bytes, kSeed);
    runtime->copyBlobFromCPU(devPtr, src.data(), bytes);

    std::vector<int8_t> dst(kDtypeElements);
    runtime->copyBlobToCPU(dst.data(), devPtr, bytes);

    int cmp = std::memcmp(src.data(), dst.data(), bytes);
    EXPECT_EQ(cmp, 0) << "INT8 round-trip mismatch";
    if (cmp == 0)
        std::cout << "[  PASSED  ] INT8 data error = 0 (" << bytes << " bytes)"
                  << std::endl;
    runtime->dealloc(devPtr);
}

} // namespace test
} // namespace infini
