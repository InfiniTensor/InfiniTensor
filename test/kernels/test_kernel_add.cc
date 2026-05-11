#include "core/graph.h"
#include "core/runtime.h"
#include "helpers/kernel_test.h"
#include "operators/element_wise.h"
#include "utils/data_generator.h"

namespace infini {
namespace test {

// Helper: build add operator on a graph
static OpBuilder makeAddBuilder() {
    return [](GraphObj *g, const TensorVec &inputs) -> Operator {
        return g->addOp<AddObj>(inputs[0], inputs[1], nullptr);
    };
}

// Helper: generate random float data using repo's DataGenerator
static std::vector<float> generateRandomData(size_t n, unsigned seed = 42) {
    RandomGenerator gen(-1.0, 1.0, seed);
    std::vector<float> data(n);
    gen(data.data(), n, DataType::Float32);
    return data;
}

// ============================================================
// CPU Golden Tests — compare against pre-computed results
// ============================================================

// [1,2] + [3,4] = [4,6]
TEST(AddCpu, Basic_1D) {
    auto result = runOpAndGetOutput(
        Device::Type::kCpu, {{{2}, {1, 2}}, {{2}, {3, 4}}}, makeAddBuilder());
    verifyGoldenData(result, {4, 6});
}

// [1,2,3] + [4,5,6] = [5,7,9]
TEST(AddCpu, Basic_1D_3elem) {
    auto result = runOpAndGetOutput(Device::Type::kCpu,
                                    {{{3}, {1, 2, 3}}, {{3}, {4, 5, 6}}},
                                    makeAddBuilder());
    verifyGoldenData(result, {5, 7, 9});
}

// [[1,2],[3,4]] + [[5,6],[7,8]] = [[6,8],[10,12]]
TEST(AddCpu, Basic_2x2) {
    auto result = runOpAndGetOutput(
        Device::Type::kCpu, {{{2, 2}, {1, 2, 3, 4}}, {{2, 2}, {5, 6, 7, 8}}},
        makeAddBuilder());
    verifyGoldenData(result, {6, 8, 10, 12});
}

// [1,2,3,4,5,6] + [-1,-2,-3,-4,-5,-6] = [0,0,0,0,0,0]
TEST(AddCpu, Negation_2x3) {
    auto result = runOpAndGetOutput(
        Device::Type::kCpu,
        {{{2, 3}, {1, 2, 3, 4, 5, 6}}, {{2, 3}, {-1, -2, -3, -4, -5, -6}}},
        makeAddBuilder());
    verifyGoldenData(result, {0, 0, 0, 0, 0, 0});
}

// Batched: [2,2,3] + [2,2,3]
TEST(AddCpu, Batched_2x2x3) {
    auto result =
        runOpAndGetOutput(Device::Type::kCpu,
                          {{{2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
                           {{2, 2, 3}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}},
                          makeAddBuilder());
    verifyGoldenData(result, {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13});
}

// ============================================================
// Cross-Platform Tests — compare against CPU results
// ============================================================

class AddCrossPlatform : public ::testing::TestWithParam<Device::Type> {};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AddCrossPlatform);

TEST_P(AddCrossPlatform, Small_128) {
    auto device = GetParam();
    auto aData = generateRandomData(128);
    auto bData = generateRandomData(128);
    verifyAgainstCpu(device, {{{128}, aData}, {{128}, bData}}, makeAddBuilder(),
                     {1e-5, 1e-5});
}

TEST_P(AddCrossPlatform, Square_128x128) {
    auto device = GetParam();
    auto aData = generateRandomData(128 * 128);
    auto bData = generateRandomData(128 * 128);
    verifyAgainstCpu(device, {{{128, 128}, aData}, {{128, 128}, bData}},
                     makeAddBuilder(), {1e-5, 1e-5});
}

TEST_P(AddCrossPlatform, Large_1024x1024) {
    auto device = GetParam();
    auto aData = generateRandomData(1024 * 1024);
    auto bData = generateRandomData(1024 * 1024);
    verifyAgainstCpu(device, {{{1024, 1024}, aData}, {{1024, 1024}, bData}},
                     makeAddBuilder(), {1e-5, 1e-5});
}

TEST_P(AddCrossPlatform, Batched_4x256x256) {
    auto device = GetParam();
    int batch = 4, H = 256, W = 256;
    auto aData = generateRandomData(batch * H * W);
    auto bData = generateRandomData(batch * H * W);
    verifyAgainstCpu(device, {{{batch, H, W}, aData}, {{batch, H, W}, bData}},
                     makeAddBuilder(), {1e-5, 1e-5});
}

// Register all non-CPU available platforms
INSTANTIATE_TEST_SUITE_P(
    AllPlatforms, AddCrossPlatform, ::testing::ValuesIn([]() {
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
// Torch Implementation Tests — verify torch Add correctness
// ============================================================

#ifdef WITH_TORCH

// Torch Add is at implementation_index = 1 in InfiniOps.
static constexpr std::size_t kTorchAddImplIndex = 1;

// CPU golden tests using torch implementation
TEST(AddTorchCpu, Basic_2x2) {
    auto result = runOpAndGetOutput(
        Device::Type::kCpu, {{{2, 2}, {1, 2, 3, 4}}, {{2, 2}, {5, 6, 7, 8}}},
        makeAddBuilder(), DataType::Float32, kTorchAddImplIndex);
    verifyGoldenData(result, {6, 8, 10, 12});
}

TEST(AddTorchCpu, Negation_2x3) {
    auto result = runOpAndGetOutput(
        Device::Type::kCpu,
        {{{2, 3}, {1, 2, 3, 4, 5, 6}}, {{2, 3}, {-1, -2, -3, -4, -5, -6}}},
        makeAddBuilder(), DataType::Float32, kTorchAddImplIndex);
    verifyGoldenData(result, {0, 0, 0, 0, 0, 0});
}

TEST(AddTorchCpu, Broadcast_2x3_plus_3) {
    auto result = runOpAndGetOutput(
        Device::Type::kCpu, {{{2, 3}, {1, 2, 3, 4, 5, 6}}, {{3}, {10, 20, 30}}},
        makeAddBuilder(), DataType::Float32, kTorchAddImplIndex);
    verifyGoldenData(result, {11, 22, 33, 14, 25, 36});
}

TEST(AddTorchCpu, Batched_2x2x3) {
    auto result = runOpAndGetOutput(
        Device::Type::kCpu,
        {{{2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
         {{2, 2, 3}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}},
        makeAddBuilder(), DataType::Float32, kTorchAddImplIndex);
    verifyGoldenData(result, {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13});
}

// Cross-platform torch tests — compare against CPU native results
class AddTorchCrossPlatform : public ::testing::TestWithParam<Device::Type> {};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AddTorchCrossPlatform);

TEST_P(AddTorchCrossPlatform, Square_128x128) {
    auto device = GetParam();
    auto aData = generateRandomData(128 * 128);
    auto bData = generateRandomData(128 * 128);
    // CPU uses native (index 0), target uses torch (index 1)
    auto cpuResult = runOpAndGetOutput(
        Device::Type::kCpu, {{{128, 128}, aData}, {{128, 128}, bData}},
        makeAddBuilder());
    auto torchResult = runOpAndGetOutput(
        device, {{{128, 128}, aData}, {{128, 128}, bData}}, makeAddBuilder(),
        DataType::Float32, kTorchAddImplIndex);
    ASSERT_EQ(cpuResult.size(), torchResult.size());
    for (size_t i = 0; i < cpuResult.size(); i++) {
        double diff = std::fabs(static_cast<double>(cpuResult[i]) -
                                static_cast<double>(torchResult[i]));
        double denom = std::max(std::fabs(static_cast<double>(cpuResult[i])),
                                std::fabs(static_cast<double>(torchResult[i])));
        double threshold = 1e-5 + 1e-5 * denom;
        EXPECT_LE(diff, threshold)
            << "Mismatch at index " << i << ": cpu=" << cpuResult[i]
            << " torch=" << torchResult[i];
    }
}

// Register all available platforms (including CPU) for torch tests
INSTANTIATE_TEST_SUITE_P(
    TorchAllPlatforms, AddTorchCrossPlatform,
    ::testing::ValuesIn([]() { return availablePlatforms(); }()),
    [](const ::testing::TestParamInfo<Device::Type> &info) {
        return "Torch_" + platformName(info.param);
    });

#endif // WITH_TORCH

} // namespace test
} // namespace infini
