#include "core/graph.h"
#include "core/runtime.h"
#include "helpers/kernel_test.h"
#include "operators/unary.h"
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
// Unary op builders — use concrete Obj types
// ============================================================

#define MAKE_UNARY_BUILDER(ObjClass)                                           \
    static OpBuilder make##ObjClass##Builder() {                               \
        return [](GraphObj *g, const TensorVec &inputs) -> Operator {          \
            return g->addOp<ObjClass>(inputs[0], nullptr);                     \
        };                                                                     \
    }

MAKE_UNARY_BUILDER(SigmoidObj)
MAKE_UNARY_BUILDER(TanhObj)
MAKE_UNARY_BUILDER(AbsObj)
MAKE_UNARY_BUILDER(ExpObj)
MAKE_UNARY_BUILDER(SqrtObj)
MAKE_UNARY_BUILDER(NegObj)
MAKE_UNARY_BUILDER(SinObj)
MAKE_UNARY_BUILDER(CosObj)
MAKE_UNARY_BUILDER(ErfObj)
MAKE_UNARY_BUILDER(CeilObj)
MAKE_UNARY_BUILDER(FloorObj)
MAKE_UNARY_BUILDER(SiluObj)
MAKE_UNARY_BUILDER(ReciprocalObj)
MAKE_UNARY_BUILDER(ASinObj)
MAKE_UNARY_BUILDER(ACosObj)
MAKE_UNARY_BUILDER(ATanObj)
MAKE_UNARY_BUILDER(SinHObj)
MAKE_UNARY_BUILDER(CosHObj)
MAKE_UNARY_BUILDER(ASinHObj)
MAKE_UNARY_BUILDER(ACosHObj)
MAKE_UNARY_BUILDER(ATanHObj)

// ============================================================
// Cross-Platform parameterized test fixture
// ============================================================

class UnaryCrossPlatform : public ::testing::TestWithParam<Device::Type> {};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(UnaryCrossPlatform);

INSTANTIATE_TEST_SUITE_P(
    AllPlatforms, UnaryCrossPlatform, ::testing::ValuesIn([]() {
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
// Sigmoid
// ============================================================
// Sigmoid
// ============================================================

TEST(UnarySigmoidCpu, Basic) {
    auto result =
        runOpAndGetOutput(Device::Type::kCpu, {{{3}, {0.0f, 1.0f, -1.0f}}},
                          makeSigmoidObjBuilder());
    EXPECT_NEAR(result[0], 0.5f, 1e-5);
    EXPECT_NEAR(result[1], 0.7311f, 1e-3);
    EXPECT_NEAR(result[2], 0.2689f, 1e-3);
}

TEST_P(UnaryCrossPlatform, Sigmoid_128) {
    auto device = GetParam();
    auto data = genData(128);
    verifyAgainstCpu(device, {{{128}, data}}, makeSigmoidObjBuilder(),
                     {1e-5, 1e-5});
}

// ============================================================
// Tanh
// ============================================================

TEST(UnaryTanhCpu, Basic) {
    auto result = runOpAndGetOutput(
        Device::Type::kCpu, {{{3}, {0.0f, 1.0f, -1.0f}}}, makeTanhObjBuilder());
    EXPECT_NEAR(result[0], 0.0f, 1e-5);
    EXPECT_NEAR(result[1], 0.7616f, 1e-3);
    EXPECT_NEAR(result[2], -0.7616f, 1e-3);
}

TEST_P(UnaryCrossPlatform, Tanh_128) {
    auto device = GetParam();
    auto data = genData(128);
    verifyAgainstCpu(device, {{{128}, data}}, makeTanhObjBuilder(),
                     {1e-5, 1e-5});
}

// ============================================================
// Abs
// ============================================================

TEST(UnaryAbsCpu, Basic) {
    auto result = runOpAndGetOutput(Device::Type::kCpu,
                                    {{{4}, {-3.0f, -0.5f, 0.0f, 2.0f}}},
                                    makeAbsObjBuilder());
    verifyGoldenData(result, {3.0f, 0.5f, 0.0f, 2.0f});
}

TEST_P(UnaryCrossPlatform, Abs_128) {
    auto device = GetParam();
    auto data = genData(128);
    verifyAgainstCpu(device, {{{128}, data}}, makeAbsObjBuilder(),
                     {1e-5, 1e-5});
}

// ============================================================
// Exp
// ============================================================

TEST(UnaryExpCpu, Basic) {
    auto result = runOpAndGetOutput(
        Device::Type::kCpu, {{{3}, {0.0f, 1.0f, 2.0f}}}, makeExpObjBuilder());
    EXPECT_NEAR(result[0], 1.0f, 1e-5);
    EXPECT_NEAR(result[1], 2.7183f, 1e-3);
    EXPECT_NEAR(result[2], 7.3891f, 1e-3);
}

TEST_P(UnaryCrossPlatform, Exp_128) {
    auto device = GetParam();
    auto data = genData(128, 99);
    verifyAgainstCpu(device, {{{128}, data}}, makeExpObjBuilder(),
                     {1e-5, 1e-5});
}

// ============================================================
// Sqrt
// ============================================================

TEST(UnarySqrtCpu, Basic) {
    auto result = runOpAndGetOutput(
        Device::Type::kCpu, {{{3}, {0.0f, 1.0f, 4.0f}}}, makeSqrtObjBuilder());
    EXPECT_NEAR(result[0], 0.0f, 1e-5);
    EXPECT_NEAR(result[1], 1.0f, 1e-5);
    EXPECT_NEAR(result[2], 2.0f, 1e-5);
}

TEST_P(UnaryCrossPlatform, Sqrt_128) {
    auto device = GetParam();
    RandomGenerator gen(0.0, 10.0, 42);
    std::vector<float> data(128);
    gen(data.data(), 128, DataType::Float32);
    verifyAgainstCpu(device, {{{128}, data}}, makeSqrtObjBuilder(),
                     {1e-5, 1e-5});
}

// ============================================================
// Neg
// ============================================================

TEST(UnaryNegCpu, Basic) {
    auto result = runOpAndGetOutput(Device::Type::kCpu,
                                    {{{4}, {-2.0f, -0.5f, 0.0f, 3.0f}}},
                                    makeNegObjBuilder());
    verifyGoldenData(result, {2.0f, 0.5f, 0.0f, -3.0f});
}

TEST_P(UnaryCrossPlatform, Neg_128) {
    auto device = GetParam();
    auto data = genData(128);
    verifyAgainstCpu(device, {{{128}, data}}, makeNegObjBuilder(),
                     {1e-5, 1e-5});
}

// ============================================================
// Sin / Cos
// ============================================================

TEST(UnarySinCpu, Basic) {
    auto result = runOpAndGetOutput(Device::Type::kCpu,
                                    {{{3}, {0.0f, 1.5708f, 3.14159f}}},
                                    makeSinObjBuilder());
    EXPECT_NEAR(result[0], 0.0f, 1e-4);
    EXPECT_NEAR(result[1], 1.0f, 1e-3);
    EXPECT_NEAR(result[2], 0.0f, 1e-3);
}

TEST(UnaryCosCpu, Basic) {
    auto result = runOpAndGetOutput(Device::Type::kCpu,
                                    {{{3}, {0.0f, 3.14159f, 6.28318f}}},
                                    makeCosObjBuilder());
    EXPECT_NEAR(result[0], 1.0f, 1e-3);
    EXPECT_NEAR(result[1], -1.0f, 1e-3);
    EXPECT_NEAR(result[2], 1.0f, 1e-3);
}

TEST_P(UnaryCrossPlatform, Sin_128) {
    auto device = GetParam();
    auto data = genData(128);
    verifyAgainstCpu(device, {{{128}, data}}, makeSinObjBuilder(),
                     {1e-5, 1e-5});
}

TEST_P(UnaryCrossPlatform, Cos_128) {
    auto device = GetParam();
    auto data = genData(128);
    verifyAgainstCpu(device, {{{128}, data}}, makeCosObjBuilder(),
                     {1e-5, 1e-5});
}

// ============================================================
// Erf
// ============================================================

TEST(UnaryErfCpu, Basic) {
    auto result = runOpAndGetOutput(Device::Type::kCpu, {{{2}, {0.0f, 1.0f}}},
                                    makeErfObjBuilder());
    EXPECT_NEAR(result[0], 0.0f, 1e-5);
    EXPECT_NEAR(result[1], 0.8427f, 1e-3);
}

TEST_P(UnaryCrossPlatform, Erf_128) {
    auto device = GetParam();
    auto data = genData(128);
    verifyAgainstCpu(device, {{{128}, data}}, makeErfObjBuilder(),
                     {1e-5, 1e-5});
}

// ============================================================
// Ceil / Floor
// ============================================================

TEST(UnaryCeilCpu, Basic) {
    auto result = runOpAndGetOutput(
        Device::Type::kCpu, {{{3}, {1.2f, -0.7f, 3.0f}}}, makeCeilObjBuilder());
    verifyGoldenData(result, {2.0f, 0.0f, 3.0f});
}

TEST(UnaryFloorCpu, Basic) {
    auto result =
        runOpAndGetOutput(Device::Type::kCpu, {{{3}, {1.8f, -0.3f, 2.0f}}},
                          makeFloorObjBuilder());
    verifyGoldenData(result, {1.0f, -1.0f, 2.0f});
}

TEST_P(UnaryCrossPlatform, Ceil_128) {
    auto device = GetParam();
    auto data = genData(128);
    verifyAgainstCpu(device, {{{128}, data}}, makeCeilObjBuilder(),
                     {1e-5, 1e-5});
}

TEST_P(UnaryCrossPlatform, Floor_128) {
    auto device = GetParam();
    auto data = genData(128);
    verifyAgainstCpu(device, {{{128}, data}}, makeFloorObjBuilder(),
                     {1e-5, 1e-5});
}

// ============================================================
// Silu
// ============================================================

TEST(UnarySiluCpu, Basic) {
    auto result = runOpAndGetOutput(Device::Type::kCpu, {{{1}, {1.0f}}},
                                    makeSiluObjBuilder());
    EXPECT_NEAR(result[0], 0.7311f, 1e-3);
}

TEST_P(UnaryCrossPlatform, Silu_128) {
    auto device = GetParam();
    auto data = genData(128);
    verifyAgainstCpu(device, {{{128}, data}}, makeSiluObjBuilder(),
                     {1e-5, 1e-5});
}

// ============================================================
// Reciprocal
// ============================================================

TEST(UnaryReciprocalCpu, Basic) {
    auto result =
        runOpAndGetOutput(Device::Type::kCpu, {{{3}, {1.0f, 2.0f, 4.0f}}},
                          makeReciprocalObjBuilder());
    EXPECT_NEAR(result[0], 1.0f, 1e-5);
    EXPECT_NEAR(result[1], 0.5f, 1e-5);
    EXPECT_NEAR(result[2], 0.25f, 1e-5);
}

TEST_P(UnaryCrossPlatform, Reciprocal_128) {
    auto device = GetParam();
    RandomGenerator gen(0.5, 5.0, 42);
    std::vector<float> data(128);
    gen(data.data(), 128, DataType::Float32);
    verifyAgainstCpu(device, {{{128}, data}}, makeReciprocalObjBuilder(),
                     {1e-5, 1e-5});
}

// ============================================================
// Inverse trig: ASin, ACos, ATan, SinH, CosH, ASinH, ACosH, ATanH
// ============================================================

TEST(UnaryASinCpu, Basic) {
    auto result = runOpAndGetOutput(Device::Type::kCpu, {{{2}, {0.0f, 0.5f}}},
                                    makeASinObjBuilder());
    EXPECT_NEAR(result[0], 0.0f, 1e-4);
    EXPECT_NEAR(result[1], 0.5236f, 1e-3);
}

TEST(UnaryACosCpu, Basic) {
    auto result = runOpAndGetOutput(Device::Type::kCpu, {{{2}, {0.0f, 1.0f}}},
                                    makeACosObjBuilder());
    EXPECT_NEAR(result[0], 1.5708f, 1e-3);
    EXPECT_NEAR(result[1], 0.0f, 1e-4);
}

TEST(UnaryATanCpu, Basic) {
    auto result = runOpAndGetOutput(Device::Type::kCpu, {{{2}, {0.0f, 1.0f}}},
                                    makeATanObjBuilder());
    EXPECT_NEAR(result[0], 0.0f, 1e-5);
    EXPECT_NEAR(result[1], 0.7854f, 1e-3);
}

TEST(UnarySinHCpu, Basic) {
    auto result = runOpAndGetOutput(Device::Type::kCpu, {{{2}, {0.0f, 1.0f}}},
                                    makeSinHObjBuilder());
    EXPECT_NEAR(result[0], 0.0f, 1e-5);
    EXPECT_NEAR(result[1], 1.1752f, 1e-3);
}

TEST(UnaryCosHCpu, Basic) {
    auto result = runOpAndGetOutput(Device::Type::kCpu, {{{2}, {0.0f, 1.0f}}},
                                    makeCosHObjBuilder());
    EXPECT_NEAR(result[0], 1.0f, 1e-5);
    EXPECT_NEAR(result[1], 1.5431f, 1e-3);
}

TEST(UnaryASinHCpu, Basic) {
    auto result = runOpAndGetOutput(Device::Type::kCpu, {{{2}, {0.0f, 1.0f}}},
                                    makeASinHObjBuilder());
    EXPECT_NEAR(result[0], 0.0f, 1e-5);
    EXPECT_NEAR(result[1], 0.8814f, 1e-3);
}

TEST(UnaryACosHCpu, Basic) {
    auto result = runOpAndGetOutput(Device::Type::kCpu, {{{2}, {1.0f, 2.0f}}},
                                    makeACosHObjBuilder());
    EXPECT_NEAR(result[0], 0.0f, 1e-5);
    EXPECT_NEAR(result[1], 1.3170f, 1e-3);
}

TEST(UnaryATanHCpu, Basic) {
    auto result = runOpAndGetOutput(Device::Type::kCpu, {{{2}, {0.0f, 0.5f}}},
                                    makeATanHObjBuilder());
    EXPECT_NEAR(result[0], 0.0f, 1e-5);
    EXPECT_NEAR(result[1], 0.5493f, 1e-3);
}

TEST_P(UnaryCrossPlatform, ASin_128) {
    auto device = GetParam();
    RandomGenerator gen(-0.9, 0.9, 42);
    std::vector<float> data(128);
    gen(data.data(), 128, DataType::Float32);
    verifyAgainstCpu(device, {{{128}, data}}, makeASinObjBuilder(),
                     {1e-5, 1e-5});
}

TEST_P(UnaryCrossPlatform, ACos_128) {
    auto device = GetParam();
    RandomGenerator gen(-0.9, 0.9, 42);
    std::vector<float> data(128);
    gen(data.data(), 128, DataType::Float32);
    verifyAgainstCpu(device, {{{128}, data}}, makeACosObjBuilder(),
                     {1e-5, 1e-5});
}

TEST_P(UnaryCrossPlatform, ATan_128) {
    auto device = GetParam();
    auto data = genData(128);
    verifyAgainstCpu(device, {{{128}, data}}, makeATanObjBuilder(),
                     {1e-5, 1e-5});
}

TEST_P(UnaryCrossPlatform, SinH_128) {
    auto device = GetParam();
    auto data = genData(128);
    verifyAgainstCpu(device, {{{128}, data}}, makeSinHObjBuilder(),
                     {1e-5, 1e-5});
}

TEST_P(UnaryCrossPlatform, CosH_128) {
    auto device = GetParam();
    auto data = genData(128);
    verifyAgainstCpu(device, {{{128}, data}}, makeCosHObjBuilder(),
                     {1e-5, 1e-5});
}

TEST_P(UnaryCrossPlatform, ASinH_128) {
    auto device = GetParam();
    auto data = genData(128);
    verifyAgainstCpu(device, {{{128}, data}}, makeASinHObjBuilder(),
                     {1e-5, 1e-5});
}

TEST_P(UnaryCrossPlatform, ACosH_128) {
    auto device = GetParam();
    RandomGenerator gen(1.1, 5.0, 42);
    std::vector<float> data(128);
    gen(data.data(), 128, DataType::Float32);
    verifyAgainstCpu(device, {{{128}, data}}, makeACosHObjBuilder(),
                     {1e-5, 1e-5});
}

TEST_P(UnaryCrossPlatform, ATanH_128) {
    auto device = GetParam();
    RandomGenerator gen(-0.9, 0.9, 42);
    std::vector<float> data(128);
    gen(data.data(), 128, DataType::Float32);
    verifyAgainstCpu(device, {{{128}, data}}, makeATanHObjBuilder(),
                     {1e-5, 1e-5});
}

} // namespace test
} // namespace infini
