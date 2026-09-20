#include "core/graph.h"
#include "core/infini_runtime.h"
#include "core/kernel.h"
#include "operators/batch_norm.h"
#include "operators/dropout.h"
#include "operators/pooling.h"
#include "test.h"
#include <cmath>
#include <cstdlib>

namespace infini {
namespace {
#if INFINITENSOR_INFINIRT_HAS_GRAPH_API
Ref<InfiniRuntimeObj> makeRuntime(size_t capacity = 2) {
    return make_ref<InfiniRuntimeObj>(
        string(::infini::rt::Device::StringFromType(
            ::infini::rt::runtime_device_type())),
        0, capacity);
}

class WorkspaceCaptureTest : public ::testing::Test {
  protected:
    void SetUp() override {
        try {
            auto runtime = makeRuntime();
            runtime->runWithGraph(make_ref<GraphObj>(runtime));
        } catch (const std::exception &error) {
            if (std::getenv("INFINITENSOR_REQUIRE_GRAPH_CAPTURE")) {
                FAIL() << error.what();
            } else {
                GTEST_SKIP() << error.what();
            }
        }
    }
};

// This test-only kernel exposes weak ownership of scratch memory. It also
// injects a capture failure after allocating workspace during warmup.
vector<std::weak_ptr<void>> scratch;
int calls = 0;
bool failCapture = false;
class WorkspaceProbeKernel : public Kernel {
    void compute(const Operator &op, const PerfRecord &,
                 const RuntimeObj *context) const override {
        compute(op, context);
    }
    void compute(const Operator &, const RuntimeObj *context) const override {
        auto runtime = dynamic_cast<const InfiniRuntimeObj *>(context);
        scratch.emplace_back(runtime->acquireWorkspace(64));
        ++calls;
        IT_ASSERT(!failCapture || calls != 2, "Injected capture failure");
    }
    PerfRecord tune(const Operator &, const RuntimeObj *) const override {
        return make_ref<PerfRecordObj>();
    }
};
[[maybe_unused]] const bool registered =
    KernelRegistry::getInstance().registerKernel(
        KernelAttrs{ExecutionProvider::Infini, OpType::Dropout},
        new WorkspaceProbeKernel(), "WorkspaceProbeKernel");

Graph probeGraph(const Ref<InfiniRuntimeObj> &runtime) {
    auto graph = make_ref<GraphObj>(runtime);
    auto input = graph->addTensor({4}, DataType::Float32);
    input->setInput();
    graph->addOp<DropoutObj>(input, nullptr, nullptr, 0.f, false);
    graph->dataMalloc();
    return graph;
}

TEST_F(WorkspaceCaptureTest, WorkspaceOwnershipAndEviction) {
    scratch.clear();
    calls = 0;
    failCapture = false;
    auto runtime = makeRuntime(1);
    auto first = probeGraph(runtime);
    runtime->runWithGraph(first);
    ASSERT_EQ(scratch.size(), 2u);
    ASSERT_FALSE(scratch[0].expired());
    EXPECT_EQ(scratch[0].lock().get(), scratch[1].lock().get());
    runtime->runWithGraph(first);
    EXPECT_EQ(calls, 2); // replay does not call the host kernel or allocate
    auto second = probeGraph(runtime);
    runtime->runWithGraph(second);
    EXPECT_TRUE(scratch[0].expired()); // LRU eviction releases storage
    EXPECT_FALSE(scratch.back().expired());
    runtime->clearGraphCache();
    EXPECT_TRUE(scratch.back().expired());
    runtime->runWithGraph(first);
    first.reset();
    second.reset();
    std::weak_ptr<InfiniRuntimeObj> weakRuntime = runtime;
    runtime.reset();
    EXPECT_TRUE(weakRuntime.expired()); // cache must not own its runtime
    EXPECT_TRUE(scratch.back().expired());
}

TEST_F(WorkspaceCaptureTest, FailureReleasesWorkspaceAndAllowsRecovery) {
    scratch.clear();
    calls = 0;
    failCapture = true;
    auto runtime = makeRuntime();
    auto graph = probeGraph(runtime);
    EXPECT_THROW(runtime->runWithGraph(graph), std::exception);
    ASSERT_EQ(scratch.size(), 2u);
    EXPECT_TRUE(scratch[0].expired());
    EXPECT_TRUE(scratch[1].expired());
    EXPECT_EQ(runtime->getGraphCacheSize(), 0u);
    failCapture = false;
    runtime->runWithGraph(graph);
    EXPECT_EQ(runtime->getGraphCaptureCount(), 1u);
    runtime->clearGraphCache();
    EXPECT_TRUE(scratch.back().expired());
    runtime->run(graph);
    EXPECT_TRUE(scratch.back().expired()); // ordinary execution is temporary
}

TEST_F(WorkspaceCaptureTest, BatchNormMaxPoolDynamicShapeAndReplay) {
#if !INFINITENSOR_TEST_HAS_ATEN
    GTEST_SKIP() << "BN/MaxPool require USE_INFINIOPS_ATEN_KERNELS";
#else
    // torch_npu's ATen provider requires Python initialization. Its BN/MaxPool
    // integration runs in test_ascend_aten.py; the two workspace ownership
    // tests above remain valid in standalone C++ on every capture backend.
    if (::infini::rt::runtime_device_type() ==
        ::infini::rt::Device::Type::kAscend)
        GTEST_SKIP() << "Ascend ATen BN/MaxPool run in test_ascend_aten.py";
    for (bool naive : {false, true}) {
        // Capacity one forces recapture when returning to an earlier shape.
        auto runtime = makeRuntime(1);
        auto graph = make_ref<GraphObj>(runtime);
        auto x = graph->addTensor({1, 2, 4, 4}, DataType::Float32);
        x->setInput();
        vector<Tensor> params;
        for (int i = 0; i < 4; ++i) {
            auto tensor = graph->addTensor({2}, DataType::Float32);
            tensor->setInput();
            params.push_back(tensor);
        }
        auto bn = graph
                      ->addOp<BatchNormObj>(x, nullptr, params[0], params[1],
                                            params[2], params[3])
                      ->getOutput();
        auto y =
            graph->addOp<MaxPoolObj>(bn, nullptr, 2, 2, 1, 1, 0, 0, 2, 2, 0)
                ->getOutput();
        y->setOutput();
        size_t captures = 0;
        for (int channels : {2, 3, 2}) {
            x->setShape({1, channels, 4, 4});
            for (auto &param : params)
                param->setShape({channels});
            graph->shape_infer();
            graph->dataMalloc(naive);
            for (int iteration = 0; iteration < 3; ++iteration) {
                vector<float> input(channels * 16), expected;
                for (size_t i = 0; i < input.size(); ++i)
                    input[i] = float(i) - 10 + iteration;
                x->copyin(input);
                params[0]->copyin(vector<float>(channels, iteration));
                params[1]->copyin(vector<float>(channels, 4));
                params[2]->copyin(vector<float>(channels, 2));
                params[3]->copyin(vector<float>(channels, -1));
                for (int c = 0; c < channels; ++c)
                    for (int h = 0; h < 4; h += 2)
                        for (int w = 0; w < 4; w += 2)
                            expected.push_back(
                                (input[c * 16 + (h + 1) * 4 + w + 1] -
                                 iteration) /
                                    std::sqrt(4.f + 1e-5f) * 2 -
                                1);
                runtime->runWithGraph(graph);
                if (iteration == 0)
                    ++captures;
                EXPECT_EQ(runtime->getGraphCaptureCount(), captures);
                EXPECT_TRUE(y->clone(NativeCpuRuntimeObj::getInstance())
                                ->equalData(expected));
            }
        }
    }
#endif
}
#endif
} // namespace
} // namespace infini
