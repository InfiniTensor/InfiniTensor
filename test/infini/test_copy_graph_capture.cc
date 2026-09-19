#include "core/graph.h"
#include "core/infini_runtime.h"
#include "operators/reshape.h"
#include "operators/rms_norm.h"
#include "operators/squeeze.h"
#include "operators/unsqueeze.h"
#include "test.h"

#include <cmath>
#include <cstdlib>
#include <tuple>

namespace infini {
namespace {

Ref<InfiniRuntimeObj> makeCopyRuntime() {
    return make_ref<InfiniRuntimeObj>(
        string(::infini::rt::Device::StringFromType(
            ::infini::rt::runtime_device_type())));
}

#if INFINITENSOR_INFINIRT_HAS_GRAPH_API
string copyGraphUnavailableReason() {
    try {
        // Probe the Graph API without invoking any operator. A broken copy
        // implementation or missing RMSNorm must not turn into a capability
        // skip. Validation on a known-capable backend can require this probe.
        auto runtime = makeCopyRuntime();
        runtime->runWithGraph(make_ref<GraphObj>(runtime));
        return {};
    } catch (const std::exception &error) {
        return error.what();
    }
}

class InfiniCopyGraphIntegrationTest : public ::testing::Test {
  protected:
    void SetUp() override {
        const auto reason = copyGraphUnavailableReason();
        if (std::getenv("INFINITENSOR_REQUIRE_GRAPH_CAPTURE")) {
            ASSERT_TRUE(reason.empty()) << reason;
        } else if (!reason.empty()) {
            GTEST_SKIP() << "Graph API unavailable: " << reason;
        }
    }
};
#endif

Tensor addCopyOp(const Graph &graph, const Tensor &input, int kind) {
    switch (kind) {
    case 0:
        return graph->addOp<ReshapeObj>(input, nullptr, Shape{3, 2})
            ->getOutput();
    case 1:
        return graph->addOp<FlattenObj>(input, nullptr, 1)->getOutput();
    case 2:
        return graph->addOp<IdentityObj>(input, nullptr)->getOutput();
    case 3:
        return graph->addOp<SqueezeObj>(input, nullptr, Shape{1})->getOutput();
    case 4:
        return graph->addOp<UnsqueezeObj>(input, nullptr, Shape{0})
            ->getOutput();
    default:
        IT_TODO_HALT();
    }
    return nullptr;
}

class InfiniCopyGraphTest
    : public ::testing::TestWithParam<std::tuple<int, bool>> {
  protected:
    void SetUp() override {
#if INFINITENSOR_INFINIRT_HAS_GRAPH_API
        const auto reason = copyGraphUnavailableReason();
        if (std::getenv("INFINITENSOR_REQUIRE_GRAPH_CAPTURE")) {
            ASSERT_TRUE(reason.empty()) << reason;
        } else if (!reason.empty()) {
            GTEST_SKIP() << "Graph API unavailable: " << reason;
        }
#endif
    }

    template <typename T> void checkCopy(DataType dtype) {
        const auto [kind, naive] = GetParam();
        auto runtime = makeCopyRuntime();
        auto graph = make_ref<GraphObj>(runtime);
        auto input = graph->addTensor({2, 1, 3}, dtype);
        input->setInput();
        auto output = addCopyOp(graph, input, kind);
        output->setOutput();
        graph->dataMalloc(naive);
        const vector<Shape> expectedShapes{
            {3, 2}, {2, 3}, {2, 1, 3}, {2, 3}, {1, 2, 1, 3}};
        ASSERT_EQ(output->getDims(), expectedShapes[kind]);
        ASSERT_EQ(output->getDType(), dtype);
        const auto inputAddress = input->getRawDataPtr<void *>();
        const auto outputAddress = output->getRawDataPtr<void *>();
        ASSERT_NE(inputAddress, outputAddress);

        vector<T> values{T(-3), T(0), T(7), T(11), T(-5), T(4)};
        input->copyin(values);
        runtime->run(graph);
        EXPECT_EQ(output->copyout<T>(), values);

#if INFINITENSOR_INFINIRT_HAS_GRAPH_API
        // Do not turn capture errors into skips: this is the regression itself.
        for (int iteration = 0; iteration < 5; ++iteration) {
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = T(iteration * 13 + int(i) - 9);
            input->copyin(values);
            runtime->runWithGraph(graph);
            EXPECT_EQ(output->copyout<T>(), values);
            EXPECT_EQ(runtime->getGraphCaptureCount(), 1u);
            EXPECT_EQ(runtime->getGraphCacheSize(), 1u);
            EXPECT_EQ(input->getRawDataPtr<void *>(), inputAddress);
            EXPECT_EQ(output->getRawDataPtr<void *>(), outputAddress);
        }
#endif
    }
};

TEST_P(InfiniCopyGraphTest, Float32OrdinaryAndReplay) {
    checkCopy<float>(DataType::Float32);
}

TEST_P(InfiniCopyGraphTest, Int32OrdinaryAndReplay) {
    checkCopy<int32_t>(DataType::Int32);
}

INSTANTIATE_TEST_SUITE_P(AllCopyOperatorsAndAllocators, InfiniCopyGraphTest,
                         ::testing::Combine(::testing::Range(0, 5),
                                            ::testing::Bool()));

TEST(InfiniCopyLifetimeTest, CloneReadbackSourceReleaseAndStorageReuse) {
    auto runtime = makeCopyRuntime();
    for (int iteration = 0; iteration < 20; ++iteration) {
        auto source =
            make_ref<TensorObj>(Shape{16384}, DataType::Int32, runtime);
        source->dataMalloc();
        vector<int32_t> values(16384);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = int32_t(i) + iteration * 16384;
        source->copyin(values);
        auto clone = source->clone(runtime);
        // Exercise the existing direct host-readback behavior without adding
        // an explicit synchronization to the caller.
        EXPECT_EQ(clone->copyout<int32_t>(), values);
        auto survivor = clone->clone(runtime);
        clone.reset();
        source.reset();
        auto reused =
            make_ref<TensorObj>(Shape{16384}, DataType::Int32, runtime);
        reused->dataMalloc();
        reused->copyin(vector<int32_t>(16384, -1));
        EXPECT_EQ(survivor->copyout<int32_t>(), values);
    }
}

// Ordinary Infini CPU copies must run even when Graph API is unavailable.
TEST(InfiniCopyLifetimeTest, OrdinaryCopyOperators) {
    auto runtime = makeCopyRuntime();
    for (int kind = 0; kind < 5; ++kind) {
        for (bool naive : {false, true}) {
            auto graph = make_ref<GraphObj>(runtime);
            auto input = graph->addTensor({2, 1, 3}, DataType::Int32);
            input->setInput();
            auto output = addCopyOp(graph, input, kind);
            output->setOutput();
            graph->dataMalloc(naive);
            vector<int32_t> values{1, -2, 3, -4, 5, 6};
            input->copyin(values);
            runtime->run(graph);
            EXPECT_EQ(output->copyout<int32_t>(), values);
        }
    }
}

#if INFINITENSOR_INFINIRT_HAS_GRAPH_API
TEST_F(InfiniCopyGraphIntegrationTest, CapturesFiveCopyNodesInSequence) {
    auto runtime = makeCopyRuntime();
    auto graph = make_ref<GraphObj>(runtime);
    auto input = graph->addTensor({2, 1, 3}, DataType::Float32);
    input->setInput();
    auto tensor =
        graph->addOp<SqueezeObj>(input, nullptr, Shape{1})->getOutput();
    tensor = graph->addOp<UnsqueezeObj>(tensor, nullptr, Shape{1})->getOutput();
    tensor =
        graph->addOp<ReshapeObj>(tensor, nullptr, Shape{3, 2})->getOutput();
    tensor = graph->addOp<FlattenObj>(tensor, nullptr, 1)->getOutput();
    auto output = graph->addOp<IdentityObj>(tensor, nullptr)->getOutput();
    output->setOutput();
    graph->dataMalloc();
    for (int iteration = 0; iteration < 5; ++iteration) {
        vector<float> values{float(iteration),     -2, 3,
                             float(iteration + 5), 0,  9};
        input->copyin(values);
        runtime->runWithGraph(graph);
        EXPECT_EQ(output->copyout<float>(), values);
        EXPECT_EQ(runtime->getGraphCaptureCount(), 1u);
    }
}

vector<float> rmsReference(vector<float> values) {
    for (size_t i = 0; i < values.size(); i += 2) {
        const float inverse =
            1.0f /
            std::sqrt((values[i] * values[i] + values[i + 1] * values[i + 1]) /
                          2.0f +
                      1e-6f);
        values[i] *= inverse;
        values[i + 1] *= inverse;
    }
    return values;
}

TEST_F(InfiniCopyGraphIntegrationTest, ComputeReshapeComputeOrdering) {
    auto runtime = makeCopyRuntime();
    auto graph = make_ref<GraphObj>(runtime);
    auto input = graph->addTensor({2, 2}, DataType::Float32);
    auto weight = graph->addTensor({2}, DataType::Float32);
    input->setInput();
    weight->setWeight();
    auto first = graph->addOp<RMSNormObj>(input, weight, nullptr)->getOutput();
    auto reshaped =
        graph->addOp<ReshapeObj>(first, nullptr, Shape{1, 2, 2})->getOutput();
    auto output =
        graph->addOp<RMSNormObj>(reshaped, weight, nullptr)->getOutput();
    output->setOutput();
    graph->dataMalloc();
    weight->copyin(vector<float>{1, 1});
    for (int iteration = 0; iteration < 5; ++iteration) {
        vector<float> values{float(iteration + 1), -2, 3, float(iteration + 4)};
        input->copyin(values);
        runtime->runWithGraph(graph);
        const auto expected = rmsReference(rmsReference(values));
        const auto actual = output->copyout<float>();
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i)
            EXPECT_NEAR(actual[i], expected[i], 1e-5f);
        EXPECT_EQ(runtime->getGraphCaptureCount(), 1u);
    }
}

TEST_F(InfiniCopyGraphIntegrationTest, CopyGraphStorageReplacementAndClear) {
    auto runtime = makeCopyRuntime();
    auto graph = make_ref<GraphObj>(runtime);
    auto input = graph->addTensor({2, 3}, DataType::Int32);
    input->setInput();
    auto output = graph->addOp<IdentityObj>(input, nullptr)->getOutput();
    output->setOutput();
    graph->dataMalloc();
    input->copyin(vector<int32_t>{1, 2, 3, 4, 5, 6});
    runtime->runWithGraph(graph);
    EXPECT_EQ(runtime->getGraphCaptureCount(), 1u);

    input->setShape({64, 3});
    graph->shape_infer();
    graph->dataMalloc();
    EXPECT_EQ(runtime->getGraphCacheSize(), 0u);
    vector<int32_t> values(192, 17);
    input->copyin(values);
    runtime->runWithGraph(graph);
    EXPECT_EQ(output->copyout<int32_t>(), values);
    EXPECT_EQ(runtime->getGraphCaptureCount(), 2u);

    runtime->clearGraphCache();
    EXPECT_EQ(runtime->getGraphCacheSize(), 0u);
    values.assign(192, -8);
    input->copyin(values);
    runtime->runWithGraph(graph);
    EXPECT_EQ(output->copyout<int32_t>(), values);
    EXPECT_EQ(runtime->getGraphCaptureCount(), 3u);
}
#endif

} // namespace
} // namespace infini
