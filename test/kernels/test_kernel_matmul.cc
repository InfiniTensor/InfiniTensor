#include "core/graph.h"
#include "core/runtime.h"
#include "helpers/kernel_test.h"
#include "operators/matmul.h"
#include "utils/data_generator.h"

namespace infini {
namespace test {

// Helper: build matmul operator on a graph
static OpBuilder makeMatmulBuilder(bool transA, bool transB) {
    return [transA, transB](GraphObj *g, const TensorVec &inputs) -> Operator {
        return g->addOp<MatmulObj>(inputs[0], inputs[1], nullptr, transA,
                                   transB);
    };
}

// Helper: generate random float data using repo's DataGenerator
static std::vector<float> generateRandomData(size_t n, unsigned seed = 42) {
    RandomGenerator gen(-1.0, 1.0, seed);
    std::vector<float> data(n);
    gen(data.data(), n, DataType::Float32);
    return data;
}

static bool matmulTransNotSupported(Device::Type device) {
    switch (device) {
    case Device::Type::kCambricon:
        return true;
    default:
        return false;
    }
}

// ============================================================
// CPU Golden Tests — compare against pre-computed results
// ============================================================

// A(2,3) @ B(3,2) = C(2,2)
// A=[[1,2,3],[4,5,6]], B=[[1,2],[3,4],[5,6]]
// C=[[22,28],[49,64]]
TEST(MatmulCpu, Basic_2x3_x_3x2) {
    auto result = runOpAndGetOutput(
        Device::Type::kCpu,
        {{{2, 3}, {1, 2, 3, 4, 5, 6}}, {{3, 2}, {1, 2, 3, 4, 5, 6}}},
        makeMatmulBuilder(false, false));
    verifyGoldenData(result, {22, 28, 49, 64});
}

// A(3,2) transA=true @ B(3,2)
// A^T=[[1,3,5],[2,4,6]], B=[[1,2],[3,4],[5,6]]
// C=[[35,44],[44,56]]
TEST(MatmulCpu, TransA_3x2_x_3x2) {
    auto result = runOpAndGetOutput(
        Device::Type::kCpu,
        {{{3, 2}, {1, 2, 3, 4, 5, 6}}, {{3, 2}, {1, 2, 3, 4, 5, 6}}},
        makeMatmulBuilder(true, false));
    verifyGoldenData(result, {35, 44, 44, 56});
}

// A(2,3) @ B(2,3) transB=true
// A=[[1,2,3],[4,5,6]], B^T=[[1,4],[2,5],[3,6]]
// C=[[14,32],[32,77]]
TEST(MatmulCpu, TransB_2x3_x_2x3) {
    auto result = runOpAndGetOutput(
        Device::Type::kCpu,
        {{{2, 3}, {1, 2, 3, 4, 5, 6}}, {{2, 3}, {1, 2, 3, 4, 5, 6}}},
        makeMatmulBuilder(false, true));
    verifyGoldenData(result, {14, 32, 32, 77});
}

// A(3,2) transA=true @ B(2,3) transB=true
// A^T=[[1,3,5],[2,4,6]], B^T=[[1,4],[2,5],[3,6]]
// C=[[22,49],[28,64]]
TEST(MatmulCpu, TransAB_3x2_x_2x3) {
    auto result = runOpAndGetOutput(
        Device::Type::kCpu,
        {{{3, 2}, {1, 2, 3, 4, 5, 6}}, {{2, 3}, {1, 2, 3, 4, 5, 6}}},
        makeMatmulBuilder(true, true));
    verifyGoldenData(result, {22, 49, 28, 64});
}

// A(3,3) @ I(3,3) = A(3,3)
// A=[[1,2,3],[4,5,6],[7,8,9]], I=[[1,0,0],[0,1,0],[0,0,1]]
TEST(MatmulCpu, Identity_3x3) {
    auto result = runOpAndGetOutput(Device::Type::kCpu,
                                    {{{3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9}},
                                     {{3, 3}, {1, 0, 0, 0, 1, 0, 0, 0, 1}}},
                                    makeMatmulBuilder(false, false));
    verifyGoldenData(result, {1, 2, 3, 4, 5, 6, 7, 8, 9});
}

// Batched A(2,2,3) @ B(2,3,2) = C(2,2,2)
TEST(MatmulCpu, Batched_2x2x3_x_2x3x2) {
    auto result = runOpAndGetOutput(
        Device::Type::kCpu,
        {{{2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
         {{2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}},
        makeMatmulBuilder(false, false));
    verifyGoldenData(result, {22, 28, 49, 64, 220, 244, 301, 334});
}

// ============================================================
// Cross-Platform Tests — compare against CPU results
// ============================================================

class MatmulCrossPlatform : public ::testing::TestWithParam<Device::Type> {};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(MatmulCrossPlatform);

TEST_P(MatmulCrossPlatform, Square_128x128) {
    auto device = GetParam();
    auto aData = generateRandomData(128 * 128);
    auto bData = generateRandomData(128 * 128);
    verifyAgainstCpu(device, {{{128, 128}, aData}, {{128, 128}, bData}},
                     makeMatmulBuilder(false, false), {1e-2, 1e-2});
}

TEST_P(MatmulCrossPlatform, Rect_128x256_x_256x128) {
    auto device = GetParam();
    auto aData = generateRandomData(128 * 256);
    auto bData = generateRandomData(256 * 128);
    verifyAgainstCpu(device, {{{128, 256}, aData}, {{256, 128}, bData}},
                     makeMatmulBuilder(false, false), {1e-2, 1e-2});
}

TEST_P(MatmulCrossPlatform, Large_1024x1024) {
    auto device = GetParam();
    auto aData = generateRandomData(1024 * 1024);
    auto bData = generateRandomData(1024 * 1024);
    verifyAgainstCpu(device, {{{1024, 1024}, aData}, {{1024, 1024}, bData}},
                     makeMatmulBuilder(false, false), {1e-2, 1.1e-2});
}

TEST_P(MatmulCrossPlatform, TransA) {
    auto device = GetParam();
    if (matmulTransNotSupported(device)) {
        GTEST_SKIP() << "transA is not supported on " << platformName(device);
    }
    int M = 128, K = 256, N = 128;
    auto aData = generateRandomData(K * M); // stored shape (K, M)
    auto bData = generateRandomData(K * N);
    verifyAgainstCpu(device, {{{K, M}, aData}, {{K, N}, bData}},
                     makeMatmulBuilder(true, false), {1e-2, 1e-2});
}

TEST_P(MatmulCrossPlatform, TransB) {
    auto device = GetParam();
    if (matmulTransNotSupported(device)) {
        GTEST_SKIP() << "transB is not supported on " << platformName(device);
    }
    int M = 128, K = 256, N = 128;
    auto aData = generateRandomData(M * K);
    auto bData = generateRandomData(N * K); // stored shape (N, K)
    verifyAgainstCpu(device, {{{M, K}, aData}, {{N, K}, bData}},
                     makeMatmulBuilder(false, true), {1e-2, 1e-2});
}

TEST_P(MatmulCrossPlatform, Batched_4x32x64) {
    auto device = GetParam();
    int batch = 4, M = 32, K = 64, N = 32;
    auto aData = generateRandomData(batch * M * K);
    auto bData = generateRandomData(batch * K * N);
    verifyAgainstCpu(device, {{{batch, M, K}, aData}, {{batch, K, N}, bData}},
                     makeMatmulBuilder(false, false), {1e-2, 1e-2});
}

// Register all non-CPU available platforms
INSTANTIATE_TEST_SUITE_P(
    AllPlatforms, MatmulCrossPlatform, ::testing::ValuesIn([]() {
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
// Torch Implementation Tests — verify torch Gemm correctness
// ============================================================

#ifdef WITH_TORCH

// Torch Gemm is at implementation_index = 2 in InfiniOps.
static constexpr std::size_t kTorchGemmImplIndex = 2;

// CPU golden tests using torch implementation
TEST(MatmulTorchCpu, Basic_2x3_x_3x2) {
    auto result = runOpAndGetOutput(
        Device::Type::kCpu,
        {{{2, 3}, {1, 2, 3, 4, 5, 6}}, {{3, 2}, {1, 2, 3, 4, 5, 6}}},
        makeMatmulBuilder(false, false), DataType::Float32,
        kTorchGemmImplIndex);
    verifyGoldenData(result, {22, 28, 49, 64});
}

TEST(MatmulTorchCpu, TransA_3x2_x_3x2) {
    auto result = runOpAndGetOutput(
        Device::Type::kCpu,
        {{{3, 2}, {1, 2, 3, 4, 5, 6}}, {{3, 2}, {1, 2, 3, 4, 5, 6}}},
        makeMatmulBuilder(true, false), DataType::Float32, kTorchGemmImplIndex);
    verifyGoldenData(result, {35, 44, 44, 56});
}

TEST(MatmulTorchCpu, TransB_2x3_x_2x3) {
    auto result = runOpAndGetOutput(
        Device::Type::kCpu,
        {{{2, 3}, {1, 2, 3, 4, 5, 6}}, {{2, 3}, {1, 2, 3, 4, 5, 6}}},
        makeMatmulBuilder(false, true), DataType::Float32, kTorchGemmImplIndex);
    verifyGoldenData(result, {14, 32, 32, 77});
}

TEST(MatmulTorchCpu, Batched_2x2x3_x_2x3x2) {
    auto result = runOpAndGetOutput(
        Device::Type::kCpu,
        {{{2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
         {{2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}},
        makeMatmulBuilder(false, false), DataType::Float32,
        kTorchGemmImplIndex);
    verifyGoldenData(result, {22, 28, 49, 64, 220, 244, 301, 334});
}

// Cross-platform torch tests — compare against CPU native results
class MatmulTorchCrossPlatform : public ::testing::TestWithParam<Device::Type> {
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(MatmulTorchCrossPlatform);

TEST_P(MatmulTorchCrossPlatform, Square_128x128) {
    auto device = GetParam();
    auto aData = generateRandomData(128 * 128);
    auto bData = generateRandomData(128 * 128);
    // CPU uses native (index 0), target uses torch (index 2)
    auto cpuResult = runOpAndGetOutput(
        Device::Type::kCpu, {{{128, 128}, aData}, {{128, 128}, bData}},
        makeMatmulBuilder(false, false));
    auto torchResult =
        runOpAndGetOutput(device, {{{128, 128}, aData}, {{128, 128}, bData}},
                          makeMatmulBuilder(false, false), DataType::Float32,
                          kTorchGemmImplIndex);
    ASSERT_EQ(cpuResult.size(), torchResult.size());
    for (size_t i = 0; i < cpuResult.size(); i++) {
        double diff = std::fabs(static_cast<double>(cpuResult[i]) -
                                static_cast<double>(torchResult[i]));
        double denom = std::max(std::fabs(static_cast<double>(cpuResult[i])),
                                std::fabs(static_cast<double>(torchResult[i])));
        double threshold = 1e-2 + 1e-2 * denom;
        EXPECT_LE(diff, threshold)
            << "Mismatch at index " << i << ": cpu=" << cpuResult[i]
            << " torch=" << torchResult[i];
    }
}

// Register all available platforms (including CPU) for torch tests
INSTANTIATE_TEST_SUITE_P(
    TorchAllPlatforms, MatmulTorchCrossPlatform,
    ::testing::ValuesIn([]() { return availablePlatforms(); }()),
    [](const ::testing::TestParamInfo<Device::Type> &info) {
        return "Torch_" + platformName(info.param);
    });

#endif // WITH_TORCH

} // namespace test
} // namespace infini
