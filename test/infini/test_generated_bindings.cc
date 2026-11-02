#include "../../src/kernels/infiniops/execution.h"
#include "core/graph.h"
#include "core/infini_runtime.h"
#include "operators/infini_op.h"
#include "operators/pooling.h"
#include "operators/reshape.h"
#include "operators/rope.h"
#include "test.h"
#include "test_infini_aten.h"
#include <cmath>
#include <map>

namespace infini {
namespace {
using Bindings = std::map<std::string, TensorVec>;
Ref<InfiniRuntimeObj> runtime() {
    initializeTestAtenProviders();
    return make_ref<InfiniRuntimeObj>(
        string(::infini::rt::Device::StringFromType(
            ::infini::rt::runtime_device_type())));
}
TEST(GeneratedInfiniOpsTest, GenericNodeClonePreservesBindings) {
    auto rt = runtime();
    auto graph = make_ref<GraphObj>(rt);
    auto x = graph->addTensor({2, 4}, DataType::Float32);
    auto w = graph->addTensor({4}, DataType::Float32);
    auto y = graph->addTensor({2, 4}, DataType::Float32);
    x->setInput();
    w->setWeight();
    y->setOutput();
    auto op = graph->addOp<InfiniOpObj>(
        "Add", Bindings{{"input", {x}}, {"other", {w}}},
        Bindings{{"out", {y}}});
    graph->dataMalloc();
    x->copyin(vector<float>(8, 2));
    w->copyin(vector<float>(4, 1));
    rt->run(graph);
    EXPECT_EQ(y->copyout<float>(), vector<float>(8, 3));
    auto clone = op->clone({x, w}, {y});
    EXPECT_EQ(clone->getOpType(), OpType::InfiniOp);
    EXPECT_EQ(clone->getInputs(), op->getInputs());
    EXPECT_EQ(clone->getOutputs(), op->getOutputs());
}
TEST(GeneratedInfiniOpsTest,
     GenericShapesAreExplicitAndValidationIsRecoverable) {
    auto rt = runtime();
    auto graph = make_ref<GraphObj>(rt);
    auto x = graph->addTensor({2, 4}, DataType::Float32);
    auto w = graph->addTensor({4}, DataType::Float32);
    auto y = graph->addTensor({2, 4}, DataType::Float32);
    auto op = graph->addOp<InfiniOpObj>(
        "Add", Bindings{{"input", {x}}, {"other", {w}}},
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
        "Add", Bindings{{"input", {intermediate}}, {"other", {w}}},
        Bindings{{"out", {y}}});
    for (int batch : {3, 1}) {
        x->setShape({batch, 4});
        op->setOutputShapes({{batch, 4}});
        graph->shape_infer();
        EXPECT_EQ(y->getDims(), (Shape{batch, 4}));
    }
}
TEST(GeneratedInfiniOpsTest, RopeReadsPositionsAfterProducerOnEveryRun) {
#if !INFINITENSOR_TEST_HAS_ATEN
    if (::infini::rt::runtime_device_type() !=
        ::infini::rt::Device::Type::kNvidia)
        GTEST_SKIP() << "Requires native NVIDIA or the ATen provider";
#endif
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
#if INFINITENSOR_TEST_HAS_ATEN
TEST(GeneratedInfiniOpsTest, PoolingSpatialChangesAndCeilBoundary) {
    auto rt = runtime();
    auto graph = make_ref<GraphObj>(rt);
    auto x = graph->addTensor({1, 1, 2, 2}, DataType::Float32);
    auto y = graph->addOp<MaxPoolObj>(x, nullptr, 1, 1, 1, 1, 0, 0, 2, 2, 1)
                 ->getOutput();
    for (int size : {2, 8, 4, 2}) {
        x->setShape({1, 1, size, size});
        graph->shape_infer();
        graph->dataMalloc();
        vector<float> input(size * size), expected;
        for (int i = 0; i < size * size; ++i)
            input[i] = i;
        for (int row = 0; row < size; row += 2)
            for (int col = 0; col < size; col += 2)
                expected.push_back(input[row * size + col]);
        x->copyin(input);
        rt->run(graph);
        EXPECT_EQ(y->copyout<float>(), expected);
    }
}

#endif
} // namespace
} // namespace infini
