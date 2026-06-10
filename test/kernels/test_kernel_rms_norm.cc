#include "core/graph.h"
#include "core/runtime.h"
#include "helpers/kernel_test.h"
#include "operators/rms_norm.h"
#include "utils/data_generator.h"
#include <cmath>

namespace infini {
namespace test {

// Helper: build RMSNorm operator on a graph
static OpBuilder makeRmsNormBuilder() {
    return [](GraphObj *g, const TensorVec &inputs) -> Operator {
        return g->addOp<RMSNormObj>(inputs[0], inputs[1], nullptr);
    };
}

// Helper: generate random float data using repo's DataGenerator
static std::vector<float> generateRandomData(size_t n, unsigned seed = 42) {
    RandomGenerator gen(-1.0, 1.0, seed);
    std::vector<float> data(n);
    gen(data.data(), n, DataType::Float32);
    return data;
}

// Helper: compute RMSNorm reference output independently.
// input shape: [batch, dim] (2D) or [batch, nhead, dim] (3D)
// weight shape: [dim]
// eps = 1e-6 (matches InfiniOps default)
static std::vector<float> computeRmsNormRef(const std::vector<float> &input,
                                            const std::vector<float> &weight,
                                            const Shape &inputShape,
                                            float eps = 1e-6f) {
    size_t ndim = inputShape.size();
    int dim = inputShape.back();
    int nhead = (ndim >= 3) ? inputShape[ndim - 2] : 1;
    int batch = (ndim >= 3) ? inputShape[ndim - 3] : inputShape[ndim - 2];

    std::vector<float> output(input.size());

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < nhead; ++h) {
            int rowOffset = b * nhead * dim + h * dim;
            float ss = 0;
            for (int k = 0; k < dim; ++k) {
                float v = input[rowOffset + k];
                ss += v * v;
            }
            float rms = 1.0f / std::sqrt(ss / static_cast<float>(dim) + eps);
            for (int k = 0; k < dim; ++k) {
                output[rowOffset + k] = input[rowOffset + k] * weight[k] * rms;
            }
        }
    }
    return output;
}

// Verify with tolerance (for golden tests where expected is computed by
// reference)
static void verifyWithTolerance(const std::vector<float> &actual,
                                const std::vector<float> &expected,
                                float rtol = 1e-5f, float atol = 1e-5f) {
    ASSERT_EQ(actual.size(), expected.size())
        << "Size mismatch: got " << actual.size() << ", expected "
        << expected.size();
    for (size_t i = 0; i < actual.size(); i++) {
        double diff = std::fabs(static_cast<double>(actual[i]) -
                                static_cast<double>(expected[i]));
        double denom = std::max(std::fabs(static_cast<double>(actual[i])),
                                std::fabs(static_cast<double>(expected[i])));
        double threshold = atol + rtol * denom;
        EXPECT_LE(diff, threshold)
            << "Mismatch at index " << i << " / " << actual.size()
            << ": actual=" << actual[i] << " expected=" << expected[i]
            << " diff=" << diff << " threshold=" << threshold;
    }
}

// ============================================================
// CPU Golden Tests — compare against reference computation
// ============================================================

// [1,4]: input=[1,2,3,4], weight=[1,1,1,1]
TEST(RmsNormCpu, Basic_1x4) {
    std::vector<float> inputData = {1, 2, 3, 4};
    std::vector<float> weightData = {1, 1, 1, 1};
    auto result = runOpAndGetOutput(Device::Type::kCpu,
                                    {{{1, 4}, inputData}, {{4}, weightData}},
                                    makeRmsNormBuilder());
    auto expected = computeRmsNormRef(inputData, weightData, {1, 4});
    verifyWithTolerance(result, expected);
}

// [1,4]: input=[2,0,0,0], weight=[0.5,1,1,1] — tests non-trivial weights
TEST(RmsNormCpu, Weighted_1x4) {
    std::vector<float> inputData = {2, 0, 0, 0};
    std::vector<float> weightData = {0.5, 1, 1, 1};
    auto result = runOpAndGetOutput(Device::Type::kCpu,
                                    {{{1, 4}, inputData}, {{4}, weightData}},
                                    makeRmsNormBuilder());
    auto expected = computeRmsNormRef(inputData, weightData, {1, 4});
    verifyWithTolerance(result, expected);
}

// [2,4]: two rows, different norms per row
TEST(RmsNormCpu, Batched_2x4) {
    std::vector<float> inputData = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> weightData = {1, 1, 1, 1};
    auto result = runOpAndGetOutput(Device::Type::kCpu,
                                    {{{2, 4}, inputData}, {{4}, weightData}},
                                    makeRmsNormBuilder());
    auto expected = computeRmsNormRef(inputData, weightData, {2, 4});
    verifyWithTolerance(result, expected);
}

// [2,2,3]: 3D input with batch and heads
TEST(RmsNormCpu, BatchedHeads_2x2x3) {
    std::vector<float> inputData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> weightData = {1, 1, 1};
    auto result = runOpAndGetOutput(Device::Type::kCpu,
                                    {{{2, 2, 3}, inputData}, {{3}, weightData}},
                                    makeRmsNormBuilder());
    auto expected = computeRmsNormRef(inputData, weightData, {2, 2, 3});
    verifyWithTolerance(result, expected);
}

// [2,3]: negative values
TEST(RmsNormCpu, NegativeValues_2x3) {
    std::vector<float> inputData = {-1, -2, -3, -4, -5, -6};
    std::vector<float> weightData = {1, 1, 1};
    auto result = runOpAndGetOutput(Device::Type::kCpu,
                                    {{{2, 3}, inputData}, {{3}, weightData}},
                                    makeRmsNormBuilder());
    auto expected = computeRmsNormRef(inputData, weightData, {2, 3});
    verifyWithTolerance(result, expected);
}

// [2,3]: weighted with mixed signs
TEST(RmsNormCpu, WeightedMixedSigns_2x3) {
    std::vector<float> inputData = {1, -2, 3, -4, 5, -6};
    std::vector<float> weightData = {0.5, 2.0, 1.5};
    auto result = runOpAndGetOutput(Device::Type::kCpu,
                                    {{{2, 3}, inputData}, {{3}, weightData}},
                                    makeRmsNormBuilder());
    auto expected = computeRmsNormRef(inputData, weightData, {2, 3});
    verifyWithTolerance(result, expected);
}

// ============================================================
// Cross-Platform Tests — compare against CPU results
// ============================================================

class RmsNormCrossPlatform : public ::testing::TestWithParam<Device::Type> {};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(RmsNormCrossPlatform);

TEST_P(RmsNormCrossPlatform, Small_1x128) {
    auto device = GetParam();
    auto inputData = generateRandomData(128);
    auto weightData = generateRandomData(128);
    verifyAgainstCpu(device, {{{1, 128}, inputData}, {{128}, weightData}},
                     makeRmsNormBuilder(), {1e-5, 1e-5});
}

TEST_P(RmsNormCrossPlatform, Batched_4x128) {
    auto device = GetParam();
    auto inputData = generateRandomData(4 * 128);
    auto weightData = generateRandomData(128);
    verifyAgainstCpu(device, {{{4, 128}, inputData}, {{128}, weightData}},
                     makeRmsNormBuilder(), {1e-5, 1e-5});
}

TEST_P(RmsNormCrossPlatform, Large_4x1024) {
    auto device = GetParam();
    auto inputData = generateRandomData(4 * 1024);
    auto weightData = generateRandomData(1024);
    verifyAgainstCpu(device, {{{4, 1024}, inputData}, {{1024}, weightData}},
                     makeRmsNormBuilder(), {1e-5, 1e-5});
}

TEST_P(RmsNormCrossPlatform, Heads_2x4x64) {
    auto device = GetParam();
    auto inputData = generateRandomData(2 * 4 * 64);
    auto weightData = generateRandomData(64);
    verifyAgainstCpu(device, {{{2, 4, 64}, inputData}, {{64}, weightData}},
                     makeRmsNormBuilder(), {1e-5, 1e-5});
}

// Register all non-CPU available platforms
INSTANTIATE_TEST_SUITE_P(
    AllPlatforms, RmsNormCrossPlatform, ::testing::ValuesIn([]() {
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
