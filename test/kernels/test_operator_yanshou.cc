#include "core/graph.h"
#include "core/runtime.h"
#include "device.h"
#include "helpers/kernel_test.h"
#include "operators/element_wise.h"
#include "operators/matmul.h"
#include "operators/rms_norm.h"
#include "operators/swiglu.h"
#include "utils/data_generator.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <iomanip>
#include <unistd.h>
#include <vector>

namespace infini {
namespace test {

static std::vector<float> genUniform(size_t n, float lo, float hi,
                                     unsigned seed = 42) {
    RandomGenerator gen(lo, hi, seed);
    std::vector<float> data(n);
    gen(data.data(), n, DataType::Float32);
    return data;
}

// ============================================================
// Test parameters: shape × device
// ============================================================

struct Shape3D {
    const char *name;
    Shape matmul_a, matmul_b; // [b, m, k], [b, k, n]
    Shape elem;               // [b, m, n]
};

struct TestParam {
    const char *shapeName;
    Shape3D shape;
    Device::Type device;
};

static const Shape3D kShapes[] = {
    {"S", {2, 8, 16}, {2, 16, 8}, {2, 8, 8}},
    {"M", {16, 64, 128}, {16, 128, 64}, {16, 64, 64}},
    {"L", {32, 256, 512}, {32, 512, 256}, {32, 256, 256}},
};

static const Device::Type kDevices[] = {
    Device::Type::kNvidia,
};

static std::vector<TestParam> buildTestParams() {
    std::vector<TestParam> params;
    for (const auto &s : kShapes) {
        for (const auto &d : kDevices) {
            params.push_back({s.name, s, d});
        }
    }
    return params;
}

// ============================================================
// Op builders
// ============================================================

static OpBuilder makeMatmulBuilder() {
    return [](GraphObj *g, const TensorVec &inputs) -> Operator {
        return g->addOp<MatmulObj>(inputs[0], inputs[1], nullptr);
    };
}

static OpBuilder makeAddBuilder() {
    return [](GraphObj *g, const TensorVec &inputs) -> Operator {
        return g->addOp<AddObj>(inputs[0], inputs[1], nullptr);
    };
}

static OpBuilder makeRMSNormBuilder() {
    return [](GraphObj *g, const TensorVec &inputs) -> Operator {
        return g->addOp<RMSNormObj>(inputs[0], inputs[1], nullptr);
    };
}

static OpBuilder makeSwiGLUBuilder() {
    return [](GraphObj *g, const TensorVec &inputs) -> Operator {
        return g->addOp<SwiGLUObj>(inputs[0], inputs[1], nullptr);
    };
}

// ============================================================
// Parameterized fixture
// ============================================================

class OperatorTest : public ::testing::TestWithParam<TestParam> {};

static std::string
testParamName(const ::testing::TestParamInfo<TestParam> &info) {
    auto devStr = Device::StringFromType(info.param.device);
    return std::string(info.param.shapeName) + "_" + std::string(devStr);
}

INSTANTIATE_TEST_SUITE_P(AllShapesAndDevices, OperatorTest,
                         ::testing::ValuesIn(buildTestParams()), testParamName);

// ============================================================
// 1. Matmul
// ============================================================

TEST_P(OperatorTest, Matmul) {
    const auto &[shapeName, cfg, device] = GetParam();
    auto nA = cfg.matmul_a[0] * cfg.matmul_a[1] * cfg.matmul_a[2];
    auto nB = cfg.matmul_b[0] * cfg.matmul_b[1] * cfg.matmul_b[2];
    auto a = genUniform(nA, -1.0f, 1.0f, 42);
    auto b = genUniform(nB, -1.0f, 1.0f, 99);
    Tolerance tol{5e-3, 1e-2};
    verifyAgainstCpu(device, {{{cfg.matmul_a, a}, {cfg.matmul_b, b}}},
                     makeMatmulBuilder(), tol);
    if (!::testing::Test::HasFailure())
        std::cout << "[  PASSED  ] Matmul/" << shapeName
                  << " meets error tolerance requirements (atol=" << tol.atol
                  << ", rtol=" << tol.rtol << ")" << std::endl;

    auto cpu = runOpAndGetOutput(Device::Type::kCpu,
                                 {{{cfg.matmul_a, a}, {cfg.matmul_b, b}}},
                                 makeMatmulBuilder());
    auto devOut = runOpAndGetOutput(
        device, {{{cfg.matmul_a, a}, {cfg.matmul_b, b}}}, makeMatmulBuilder());
}

// ============================================================
// 2. Add (element-wise)
// ============================================================

TEST_P(OperatorTest, Add) {
    const auto &[shapeName, cfg, device] = GetParam();
    auto n = cfg.elem[0] * cfg.elem[1] * cfg.elem[2];
    auto a = genUniform(n, -1.0f, 1.0f, 42);
    auto b = genUniform(n, -1.0f, 1.0f, 99);
    Tolerance tol{1e-5, 1e-5};
    verifyAgainstCpu(device, {{{cfg.elem, a}, {cfg.elem, b}}}, makeAddBuilder(), tol);
    if (!::testing::Test::HasFailure())
        std::cout << "[  PASSED  ] Add/" << shapeName
                  << " meets error tolerance requirements (atol=" << tol.atol
                  << ", rtol=" << tol.rtol << ")" << std::endl;
    auto cpu = runOpAndGetOutput(
        Device::Type::kCpu, {{{cfg.elem, a}, {cfg.elem, b}}}, makeAddBuilder());
    auto devOut = runOpAndGetOutput(device, {{{cfg.elem, a}, {cfg.elem, b}}},
                                    makeAddBuilder());
}

// ============================================================
// 3. RMSNorm — input [B, M, K], weight [K]
// ============================================================

static Shape rmsnormWeight(const Shape &input) { return {input.back()}; }

TEST_P(OperatorTest, RMSNorm) {
    const auto &[shapeName, cfg, device] = GetParam();
    auto nInput = cfg.matmul_a[0] * cfg.matmul_a[1] * cfg.matmul_a[2];
    auto weightShape = rmsnormWeight(cfg.matmul_a);
    auto nWeight = weightShape[0];
    auto input = genUniform(nInput, -1.0f, 1.0f, 42);
    auto weight = genUniform(nWeight, 0.5f, 1.5f, 99);
    Tolerance tol{1e-3, 1e-3};
    verifyAgainstCpu(device, {{{cfg.matmul_a, input}, {weightShape, weight}}},
                     makeRMSNormBuilder(), tol);
    if (!::testing::Test::HasFailure())
        std::cout << "[  PASSED  ] RMSNorm/" << shapeName
                  << " meets error tolerance requirements (atol=" << tol.atol
                  << ", rtol=" << tol.rtol << ")" << std::endl;
    auto cpu = runOpAndGetOutput(
        Device::Type::kCpu, {{{cfg.matmul_a, input}, {weightShape, weight}}},
        makeRMSNormBuilder());
    auto devOut = runOpAndGetOutput(
        device, {{{cfg.matmul_a, input}, {weightShape, weight}}},
        makeRMSNormBuilder());
}

// ============================================================
// 4. SwiGLU — input [B, M, K], gate [B, M, K]
// ============================================================

TEST_P(OperatorTest, SwiGLU) {
    const auto &[shapeName, cfg, device] = GetParam();
    auto n = cfg.matmul_a[0] * cfg.matmul_a[1] * cfg.matmul_a[2];
    auto input = genUniform(n, -1.0f, 1.0f, 42);
    auto gate = genUniform(n, -1.0f, 1.0f, 99);
    Tolerance tol{1e-4, 1e-4};
    verifyAgainstCpu(device, {{{cfg.matmul_a, input}, {cfg.matmul_a, gate}}},
                     makeSwiGLUBuilder(), tol);
    if (!::testing::Test::HasFailure())
        std::cout << "[  PASSED  ] SwiGLU/" << shapeName
                  << " meets error tolerance requirements (atol=" << tol.atol
                  << ", rtol=" << tol.rtol << ")" << std::endl;
    auto cpu = runOpAndGetOutput(
        Device::Type::kCpu, {{{cfg.matmul_a, input}, {cfg.matmul_a, gate}}},
        makeSwiGLUBuilder());
    auto devOut = runOpAndGetOutput(
        device, {{{cfg.matmul_a, input}, {cfg.matmul_a, gate}}},
        makeSwiGLUBuilder());
}

} // namespace test
} // namespace infini
