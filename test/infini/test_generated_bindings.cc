#include "../../src/kernels/infiniops/execution.h"
#include "core/graph.h"
#include "core/infini_runtime.h"
#include "operators/infini_op.h"
#include "operators/reshape.h"
#include "operators/rope.h"
#include "operators/unary.h"
#include "test.h"
#include <cmath>
#include <map>

namespace infini {
namespace {
using Bindings = std::map<std::string, TensorVec>;
Ref<InfiniRuntimeObj> runtime() {
    return make_ref<InfiniRuntimeObj>(
        string(::infini::rt::Device::StringFromType(
            ::infini::rt::runtime_device_type())));
}
TEST(GeneratedInfiniOpsTest, GenericNodeWithNamedAttributesAndClone) {
    auto rt = runtime();
    auto graph = make_ref<GraphObj>(rt);
    auto x = graph->addTensor({2, 4}, DataType::Float32);
    auto w = graph->addTensor({4}, DataType::Float32);
    auto y = graph->addTensor({2, 4}, DataType::Float32);
    x->setInput();
    w->setWeight();
    y->setOutput();
    auto op = graph->addOp<InfiniOpObj>(
        "RmsNorm", Bindings{{"input", {x}}, {"weight", {w}}},
        Bindings{{"out", {y}}}, nlohmann::json{{"eps", 1e-6}});
    graph->dataMalloc();
    x->copyin(vector<float>{1, 2, 3, 4, -1, -2, -3, -4});
    w->copyin(vector<float>{1, 1, 1, 1});
    rt->run(graph);
    vector<float> expected;
    for (float v : vector<float>{1, 2, 3, 4, -1, -2, -3, -4})
        expected.push_back(v / std::sqrt(7.5f + 1e-6f));
    EXPECT_TRUE(
        y->clone(NativeCpuRuntimeObj::getInstance())->equalData(expected));
    auto old = y->copyout<float>();
    const auto saved = old;
    auto clone = op->clone({x, w}, {y});
    EXPECT_EQ(clone->getOpType(), OpType::InfiniOp);
    // No new C++ kernel, enum entry or factory mapping is needed for this node.
    if (::infini::rt::runtime_device_type() ==
        ::infini::rt::Device::Type::kNvidia) {
        rt->runWithGraph(graph);
        rt->runWithGraph(graph);
        EXPECT_EQ(rt->getGraphCaptureCount(), 1u);
        EXPECT_TRUE(
            y->clone(NativeCpuRuntimeObj::getInstance())->equalData(expected));
        rt->clearGraphCache();
    }
    x->copyin(vector<float>(8, 0));
    rt->run(graph);
    EXPECT_EQ(old, saved);
    EXPECT_EQ(y->copyout<float>(), vector<float>(8, 0));
}
TEST(GeneratedInfiniOpsTest,
     GenericShapesAreExplicitAndValidationIsRecoverable) {
    auto rt = runtime();
    auto graph = make_ref<GraphObj>(rt);
    auto x = graph->addTensor({2, 4}, DataType::Float32);
    auto w = graph->addTensor({4}, DataType::Float32);
    auto y = graph->addTensor({2, 4}, DataType::Float32);
    auto op = graph->addOp<InfiniOpObj>(
        "RmsNorm", Bindings{{"input", {x}}, {"weight", {w}}},
        Bindings{{"out", {y}}});
    x->setShape({3, 4});
    EXPECT_THROW(graph->shape_infer(), std::exception);
    op->setOutputShapes({{3, 4}});
    graph->shape_infer();
    EXPECT_EQ(y->getDims(), (Shape{3, 4}));
    EXPECT_THROW(op->setOutputShapes({}), std::exception);
}
TEST(GeneratedInfiniOpsTest, ExplicitShapesFollowIntermediateInputs) {
    auto rt = runtime();
    auto graph = make_ref<GraphObj>(rt);
    auto x = graph->addTensor({1, 4}, DataType::Float32);
    auto w = graph->addTensor({4}, DataType::Float32);
    auto intermediate = graph->addOp<IdentityObj>(x, nullptr)->getOutput();
    auto y = graph->addTensor({1, 4}, DataType::Float32);
    auto op = graph->addOp<InfiniOpObj>(
        "RmsNorm", Bindings{{"input", {intermediate}}, {"weight", {w}}},
        Bindings{{"out", {y}}});
    for (int batch : {3, 1}) {
        x->setShape({batch, 4});
        op->setOutputShapes({{batch, 4}});
        graph->shape_infer();
        EXPECT_EQ(y->getDims(), (Shape{batch, 4}));
    }
}
TEST(GeneratedInfiniOpsTest,
     WrongNamesTypesAndMissingBindingsFailBeforeExecution) {
    auto rt = runtime();
    auto graph = make_ref<GraphObj>(rt);
    auto x = graph->addTensor({2, 4}, DataType::Float32);
    auto w = graph->addTensor({4}, DataType::Float32);
    auto y = graph->addTensor({2, 4}, DataType::Float32);
    x->setInput();
    w->setInput();
    y->setOutput();
    graph->addOp<InfiniOpObj>(
        "RmsNorm", Bindings{{"input", {x}}, {"weight", {w}}},
        Bindings{{"out", {y}}}, nlohmann::json{{"eps", true}});
    graph->dataMalloc();
    x->copyin(vector<float>(8, 1));
    w->copyin(vector<float>(4, 1));
    y->copyin(vector<float>(8, 77));
    EXPECT_THROW(rt->run(graph), std::exception);
    EXPECT_EQ(y->copyout<float>(), vector<float>(8, 77));
    EXPECT_THROW(graph->addOp<InfiniOpObj>("RmsNorm", Bindings{{"out", {x}}},
                                           Bindings{{"out", {y}}}),
                 std::exception);
}
#if INFINITENSOR_TEST_HAS_ATEN
TEST(GeneratedInfiniOpsTest, RopeReadsPositionsAfterProducerOnEveryRun) {
    auto rt = runtime();
    auto graph = make_ref<GraphObj>(rt);
    auto positions = graph->addTensor({1, 2}, DataType::Int64);
    auto x = graph->addTensor({1, 2, 128}, DataType::Float32);
    positions->setInput();
    x->setInput();
    auto produced = graph->addOp<IdentityObj>(positions, nullptr)->getOutput();
    auto y = graph->addOp<RoPEObj>(produced, x, nullptr)->getOutput();
    y->setOutput();
    graph->dataMalloc();
    x->copyin(vector<float>(256, 1));
    for (int64_t position : {3, 7, 1}) {
        positions->copyin(vector<int64_t>{position, position + 1});
        rt->run(graph);
        auto result = y->copyout<float>();
        for (int token = 0; token < 2; ++token) {
            for (int feature = 0; feature < 64; ++feature) {
                const float angle =
                    (position + token) *
                    std::pow(10000.0f, -2.0f * feature / 128.0f);
                EXPECT_NEAR(result[token * 128 + feature],
                            std::cos(angle) - std::sin(angle), 1e-5);
                EXPECT_NEAR(result[token * 128 + feature + 64],
                            std::cos(angle) + std::sin(angle), 1e-5);
            }
        }
    }
    EXPECT_THROW(rt->runWithGraph(graph), std::exception);
    rt->run(graph);
}
TEST(GeneratedInfiniOpsTest, AddedSdkOperatorsRequireNoHandwrittenBridge) {
    const auto schemas =
        nlohmann::json::parse(infiniops::Registry::instance().schemas());
    ASSERT_TRUE(schemas.contains("Abs"));
    ASSERT_TRUE(schemas.contains("Erf"));
    auto rt = runtime();
    for (const auto &name : {"Abs", "Erf"}) {
        auto graph = make_ref<GraphObj>(rt);
        auto x = graph->addTensor({2, 3}, DataType::Float32);
        auto y = graph->addTensor({2, 3}, DataType::Float32);
        x->setInput();
        y->setOutput();
        graph->addOp<InfiniOpObj>(name, Bindings{{"input", {x}}},
                                  Bindings{{"out", {y}}});
        graph->dataMalloc();
        const vector<float> values{-2, -1, 0, 0.5, 1, 2};
        x->copyin(values);
        rt->run(graph);
        vector<float> expected;
        for (float v : values)
            expected.push_back(string(name) == "Abs" ? std::abs(v)
                                                     : std::erf(v));
        EXPECT_TRUE(
            y->clone(NativeCpuRuntimeObj::getInstance())->equalData(expected));
        EXPECT_THROW(rt->runWithGraph(graph), std::exception);
        rt->run(graph);
        EXPECT_TRUE(
            y->clone(NativeCpuRuntimeObj::getInstance())->equalData(expected));
    }
    auto graph = make_ref<GraphObj>(rt);
    auto x = graph->addTensor({2, 3}, DataType::Float32);
    auto y = graph->addOp<AbsObj>(x, nullptr)->getOutput();
    graph->dataMalloc();
    x->copyin(vector<float>{-2, -1, 0, 0.5, 1, 2});
    rt->run(graph);
    EXPECT_EQ(y->copyout<float>(), (vector<float>{2, 1, 0, 0.5, 1, 2}));
}
#endif
} // namespace
} // namespace infini
