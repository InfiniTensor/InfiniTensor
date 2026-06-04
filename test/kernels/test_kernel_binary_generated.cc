#include "core/graph.h"
#include "core/runtime.h"
#include "helpers/kernel_test.h"
#include "operators/element_wise.h"
#include "utils/data_generator.h"

namespace infini {
namespace test {

static std::vector<float> genData(size_t n, unsigned seed = 42) {
    RandomGenerator gen(-1.0, 1.0, seed);
    std::vector<float> data(n);
    gen(data.data(), n, DataType::Float32);
    return data;
}

// ============================================================
// Binary op builders
// ============================================================

static OpBuilder makeDivBuilder() {
    return [](GraphObj *g, const TensorVec &inputs) -> Operator {
        return g->addOp<DivObj>(inputs[0], inputs[1], nullptr);
    };
}

static OpBuilder makeMaxBuilder() {
    return [](GraphObj *g, const TensorVec &inputs) -> Operator {
        return g->addOp<MaximumObj>(inputs[0], inputs[1], nullptr);
    };
}

static OpBuilder makeMinBuilder() {
    return [](GraphObj *g, const TensorVec &inputs) -> Operator {
        return g->addOp<MinimumObj>(inputs[0], inputs[1], nullptr);
    };
}

// ============================================================
// Cross-Platform parameterized test fixture
// ============================================================

class BinaryGenCrossPlatform : public ::testing::TestWithParam<Device::Type> {};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(BinaryGenCrossPlatform);

INSTANTIATE_TEST_SUITE_P(
    AllPlatforms, BinaryGenCrossPlatform, ::testing::ValuesIn([]() {
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

// ============================================================
// Div tests
// ============================================================

TEST(BinaryDivCpu, Basic) {
    auto result = runOpAndGetOutput(
        Device::Type::kCpu,
        {{{3}, {6.0f, 3.0f, 1.0f}}, {{3}, {2.0f, 3.0f, 1.0f}}},
        makeDivBuilder());
    EXPECT_NEAR(result[0], 3.0f, 1e-5);
    EXPECT_NEAR(result[1], 1.0f, 1e-5);
    EXPECT_NEAR(result[2], 1.0f, 1e-5);
}

TEST(BinaryDivCpu, Negative) {
    auto result = runOpAndGetOutput(
        Device::Type::kCpu, {{{2}, {-6.0f, 6.0f}}, {{2}, {3.0f, -2.0f}}},
        makeDivBuilder());
    EXPECT_NEAR(result[0], -2.0f, 1e-5);
    EXPECT_NEAR(result[1], -3.0f, 1e-5);
}

TEST_P(BinaryGenCrossPlatform, Div_128) {
    auto device = GetParam();
    RandomGenerator gen(0.5, 5.0, 42);
    std::vector<float> a(128), b(128);
    gen(a.data(), 128, DataType::Float32);
    gen(b.data(), 128, DataType::Float32);
    verifyAgainstCpu(device, {{{128}, a}, {{128}, b}}, makeDivBuilder(),
                     {1e-5, 1e-5});
}

TEST_P(BinaryGenCrossPlatform, Div_8x8) {
    auto device = GetParam();
    RandomGenerator gen(0.5, 5.0, 42);
    std::vector<float> a(64), b(64);
    gen(a.data(), 64, DataType::Float32);
    gen(b.data(), 64, DataType::Float32);
    verifyAgainstCpu(device, {{{8, 8}, a}, {{8, 8}, b}}, makeDivBuilder(),
                     {1e-5, 1e-5});
}

// ============================================================
// Max tests
// ============================================================

TEST(BinaryMaxCpu, Basic) {
    auto result = runOpAndGetOutput(
        Device::Type::kCpu,
        {{{4}, {1.0f, 5.0f, 3.0f, -2.0f}}, {{4}, {2.0f, 3.0f, 3.0f, 1.0f}}},
        makeMaxBuilder());
    verifyGoldenData(result, {2.0f, 5.0f, 3.0f, 1.0f});
}

TEST(BinaryMaxCpu, Mixed) {
    auto result = runOpAndGetOutput(
        Device::Type::kCpu,
        {{{3}, {-1.0f, 0.0f, 1.0f}}, {{3}, {1.0f, 0.0f, -1.0f}}},
        makeMaxBuilder());
    verifyGoldenData(result, {1.0f, 0.0f, 1.0f});
}

TEST_P(BinaryGenCrossPlatform, Max_128) {
    auto device = GetParam();
    auto a = genData(128, 42);
    auto b = genData(128, 99);
    verifyAgainstCpu(device, {{{128}, a}, {{128}, b}}, makeMaxBuilder(),
                     {1e-5, 1e-5});
}

// ============================================================
// Min tests
// ============================================================

TEST(BinaryMinCpu, Basic) {
    auto result = runOpAndGetOutput(
        Device::Type::kCpu,
        {{{4}, {1.0f, 5.0f, 3.0f, -2.0f}}, {{4}, {2.0f, 3.0f, 3.0f, 1.0f}}},
        makeMinBuilder());
    verifyGoldenData(result, {1.0f, 3.0f, 3.0f, -2.0f});
}

TEST(BinaryMinCpu, Mixed) {
    auto result = runOpAndGetOutput(
        Device::Type::kCpu,
        {{{3}, {-1.0f, 0.0f, 1.0f}}, {{3}, {1.0f, 0.0f, -1.0f}}},
        makeMinBuilder());
    verifyGoldenData(result, {-1.0f, 0.0f, -1.0f});
}

TEST_P(BinaryGenCrossPlatform, Min_128) {
    auto device = GetParam();
    auto a = genData(128, 42);
    auto b = genData(128, 99);
    verifyAgainstCpu(device, {{{128}, a}, {{128}, b}}, makeMinBuilder(),
                     {1e-5, 1e-5});
}

} // namespace test
} // namespace infini
