#include "core/graph.h"
#include "core/runtime.h"
#include "helpers/kernel_test.h"
#include "operators/swiglu.h"
#include "utils/data_generator.h"

namespace infini {
namespace test {

static OpBuilder makeSwiGLUBuilder() {
    return [](GraphObj *g, const TensorVec &inputs) -> Operator {
        return g->addOp<SwiGLUObj>(inputs[0], inputs[1], nullptr);
    };
}

// SwiGLU(input, gate) = input * silu(gate) = input * gate * sigmoid(gate)
// gate=1.0: silu(1.0) = 1.0 * sigmoid(1.0) ≈ 0.7311
// input=[2,3]: output ≈ [1.4622, 2.1933]
TEST(SwiGLUCpu, Basic) {
    auto result = runOpAndGetOutput(Device::Type::kCpu,
                                    {{{2}, {2.0f, 3.0f}}, {{2}, {1.0f, 1.0f}}},
                                    makeSwiGLUBuilder());
    float silu_1 = 1.0f / (1.0f + std::exp(-1.0f)); // sigmoid(1.0)
    silu_1 *= 1.0f; // silu(1.0) = gate * sigmoid(gate)
    verifyGoldenData(result, {2.0f * silu_1, 3.0f * silu_1});
}

// gate=0.0: silu(0.0) = 0.0 * sigmoid(0.0) = 0.0 * 0.5 = 0.0
// input=[5,-3]: output = [0.0, 0.0]
TEST(SwiGLUCpu, ZeroGate) {
    auto result = runOpAndGetOutput(Device::Type::kCpu,
                                    {{{2}, {5.0f, -3.0f}}, {{2}, {0.0f, 0.0f}}},
                                    makeSwiGLUBuilder());
    verifyGoldenData(result, {0.0f, 0.0f});
}

// gate=2.0: silu(2.0) = 2.0 * sigmoid(2.0) ≈ 1.7616
// input=[1,1,1,1]: output ≈ [1.7616, 1.7616, 1.7616, 1.7616]
TEST(SwiGLUCpu, UniformGate) {
    auto result = runOpAndGetOutput(
        Device::Type::kCpu,
        {{{4}, {1.0f, 1.0f, 1.0f, 1.0f}}, {{4}, {2.0f, 2.0f, 2.0f, 2.0f}}},
        makeSwiGLUBuilder());
    float silu_2 = 2.0f / (1.0f + std::exp(-2.0f));
    verifyGoldenData(result, {silu_2, silu_2, silu_2, silu_2});
}

// ============================================================
// Cross-Platform Tests — compare against CPU results
// ============================================================

static std::vector<float> generateRandomData(size_t n, unsigned seed = 42) {
    RandomGenerator gen(-1.0, 1.0, seed);
    std::vector<float> data(n);
    gen(data.data(), n, DataType::Float32);
    return data;
}

class SwiGLUCrossPlatform : public ::testing::TestWithParam<Device::Type> {};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(SwiGLUCrossPlatform);

TEST_P(SwiGLUCrossPlatform, Small_128) {
    auto device = GetParam();
    auto aData = generateRandomData(128);
    auto bData = generateRandomData(128);
    verifyAgainstCpu(device, {{{128}, aData}, {{128}, bData}},
                     makeSwiGLUBuilder(), {1e-5, 1e-5});
}

TEST_P(SwiGLUCrossPlatform, Square_128x128) {
    auto device = GetParam();
    auto aData = generateRandomData(128 * 128);
    auto bData = generateRandomData(128 * 128);
    verifyAgainstCpu(device, {{{128, 128}, aData}, {{128, 128}, bData}},
                     makeSwiGLUBuilder(), {1e-5, 1e-5});
}

TEST_P(SwiGLUCrossPlatform, Large_1024x1024) {
    auto device = GetParam();
    auto aData = generateRandomData(1024 * 1024);
    auto bData = generateRandomData(1024 * 1024);
    verifyAgainstCpu(device, {{{1024, 1024}, aData}, {{1024, 1024}, bData}},
                     makeSwiGLUBuilder(), {1e-5, 1e-5});
}

TEST_P(SwiGLUCrossPlatform, Batched_4x256x256) {
    auto device = GetParam();
    int batch = 4, H = 256, W = 256;
    auto aData = generateRandomData(batch * H * W);
    auto bData = generateRandomData(batch * H * W);
    verifyAgainstCpu(device, {{{batch, H, W}, aData}, {{batch, H, W}, bData}},
                     makeSwiGLUBuilder(), {1e-5, 1e-5});
}

INSTANTIATE_TEST_SUITE_P(
    AllPlatforms, SwiGLUCrossPlatform, ::testing::ValuesIn([]() {
        auto all = availablePlatforms();
        std::vector<Device::Type> nonCpu;
        for (auto p : all) {
            if (p != Device::Type::kCpu) {
                nonCpu.push_back(p);
            }
        }
        return nonCpu;
    }()),
    [](const ::testing::TestParamInfo<Device::Type> &info) {
        return platformName(info.param);
    });

} // namespace test
} // namespace infini
