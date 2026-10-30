#include "../../src/kernels/infiniops/execution.h"
#include "core/graph.h"
#include "core/infini_runtime.h"
#include "operators/infini_op.h"
#include "operators/reshape.h"
#include "operators/rope.h"
#include "operators/unary.h"
#include "test.h"
#include "test_infini_aten.h"
#include <cmath>
#include <map>
#include <thread>

namespace infini {
namespace {
using Bindings = std::map<std::string, TensorVec>;
Ref<InfiniRuntimeObj> runtime() {
    initializeTestAtenProviders();
    return make_ref<InfiniRuntimeObj>(
        string(::infini::rt::Device::StringFromType(
            ::infini::rt::runtime_device_type())));
}
#if INFINITENSOR_TEST_HAS_ATEN
TEST(GeneratedInfiniOpsTest, UninitializedAscendAtenFailsWithoutCrashing) {
    if (::infini::rt::runtime_device_type() !=
        ::infini::rt::Device::Type::kAscend)
        GTEST_SKIP() << "Requires the Ascend ATen provider";
    ASSERT_FALSE(Py_IsInitialized());
    auto rt = make_ref<InfiniRuntimeObj>("ascend");
    auto graph = make_ref<GraphObj>(rt);
    auto x = graph->addTensor({1}, DataType::Float32);
    auto y = graph->addTensor({1}, DataType::Float32);
    graph->addOp<InfiniOpObj>("Erf", Bindings{{"input", {x}}},
                              Bindings{{"out", {y}}});
    graph->dataMalloc();
    x->copyin(vector<float>{1});
    y->copyin(vector<float>{77});
    try {
        rt->run(graph);
        FAIL() << "Uninitialized provider must be rejected";
    } catch (const std::exception &error) {
        EXPECT_NE(string(error.what()).find("initialized Python"),
                  string::npos);
    }
    EXPECT_EQ(y->copyout<float>(), vector<float>{77});
}
#endif
TEST(GeneratedInfiniOpsTest, NvidiaGemmCanMoveFromWarmupToWorker) {
    if (::infini::rt::runtime_device_type() !=
            ::infini::rt::Device::Type::kNvidia ||
        infiniops::Registry::instance().schemas().find("Gemm") == string::npos)
        GTEST_SKIP() << "Requires an SDK with native NVIDIA Gemm";
    auto rt = runtime();
    auto graph = make_ref<GraphObj>(rt);
    auto a = graph->addTensor({2, 3}, DataType::Float32);
    auto b = graph->addTensor({3, 4}, DataType::Float32);
    auto y = graph->addTensor({2, 4}, DataType::Float32);
    a->setInput();
    b->setInput();
    y->setOutput();
    graph->addOp<InfiniOpObj>("Gemm", Bindings{{"a", {a}}, {"b", {b}}},
                              Bindings{{"y", {y}}});
    graph->dataMalloc();
    a->copyin(vector<float>(6, 1));
    b->copyin(vector<float>(12, 1));
    rt->run(graph);
    std::exception_ptr error;
    std::thread worker([&] {
        try {
            rt->run(graph);
            rt->runWithGraph(graph);
        } catch (...) {
            error = std::current_exception();
        }
    });
    worker.join();
    if (error) {
        EXPECT_NO_THROW(std::rethrow_exception(error));
    }
    EXPECT_EQ(y->copyout<float>(), vector<float>(8, 3));
    EXPECT_EQ(rt->getGraphCaptureCount(), 1u);
    rt->runWithGraph(graph);
    rt->clearGraphCache();
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
    for (auto positionType : {DataType::Int64, DataType::Int32}) {
        auto graph = make_ref<GraphObj>(rt);
        auto positions = graph->addTensor({1, 2}, positionType);
        auto x = graph->addTensor({1, 2, 128}, DataType::Float32);
        positions->setInput();
        x->setInput();
        auto produced =
            graph->addOp<IdentityObj>(positions, nullptr)->getOutput();
        auto y = graph->addOp<RoPEObj>(produced, x, nullptr)->getOutput();
        y->setOutput();
        graph->dataMalloc();
        vector<float> values(256);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = (int(i % 23) - 11) / 16.f;
        x->copyin(values);
        for (int64_t position : {3, 7, 1}) {
            if (positionType == DataType::Int64)
                positions->copyin(vector<int64_t>{position, position + 1});
            else
                positions->copyin(
                    vector<int32_t>{int32_t(position), int32_t(position + 1)});
            rt->run(graph);
            auto result = y->copyout<float>();
            EXPECT_EQ(x->copyout<float>(), values);
            for (int token = 0; token < 2; ++token) {
                for (int feature = 0; feature < 64; ++feature) {
                    const float angle =
                        (position + token) *
                        std::pow(10000.0f, -2.0f * feature / 128.0f);
                    EXPECT_NEAR(result[token * 128 + feature],
                                values[token * 128 + feature] *
                                        std::cos(angle) -
                                    values[token * 128 + feature + 64] *
                                        std::sin(angle),
                                1e-5);
                    EXPECT_NEAR(
                        result[token * 128 + feature + 64],
                        values[token * 128 + feature + 64] * std::cos(angle) +
                            values[token * 128 + feature] * std::sin(angle),
                        1e-5);
                }
            }
        }
        EXPECT_THROW(rt->runWithGraph(graph), std::exception);
        rt->run(graph);
    }
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
        auto actual = y->copyout<float>();
        for (size_t i = 0; i < expected.size(); ++i)
            EXPECT_NEAR(actual[i], expected[i],
                        string(name) == "Abs" ? 0.f : 1e-5f);
        EXPECT_THROW(rt->runWithGraph(graph), std::exception);
        rt->run(graph);
        actual = y->copyout<float>();
        for (size_t i = 0; i < expected.size(); ++i)
            EXPECT_NEAR(actual[i], expected[i],
                        string(name) == "Abs" ? 0.f : 1e-5f);
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
