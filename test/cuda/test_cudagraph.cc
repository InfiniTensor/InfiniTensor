#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "operators/dropout.h"
#include "operators/matmul.h"
#include "operators/unary.h"

#include "test.h"
#include <algorithm>
#include <atomic>
#include <thread>

namespace infini {

namespace {
class CaptureUnsupportedKernel : public CudaKernelWithoutConfig {
    void compute(const Operator &, const RuntimeObj *) const override {
        checkCudaError(cudaStreamSynchronize(CUDAStream::getCurrentStream()));
    }
};

[[maybe_unused]] const bool captureUnsupportedKernelRegistered =
    KernelRegistry::getInstance().registerKernel(
        KernelAttrs{Device::CUDA, OpType::Dropout},
        new CaptureUnsupportedKernel(), "CaptureUnsupportedKernel");

class CudaGraphFixture {
  public:
    Ref<CudaRuntimeObj> runtime;
    Graph graph;
    Tensor input;
    Tensor weight;
    Tensor output;

    explicit CudaGraphFixture(Ref<CudaRuntimeObj> runtime, int batch = 8)
        : runtime(std::move(runtime)),
          graph(make_ref<GraphObj>(this->runtime)) {
        input = graph->addTensor({batch, 2}, DataType::Float32);
        weight = graph->addTensor({2, 2}, DataType::Float32);
        input->setInput();
        weight->setWeight();
        auto matmul = graph->addOp<MatmulObj>(input, weight, nullptr);
        output =
            graph->addOp<ReluObj>(matmul->getOutput(), nullptr)->getOutput();
        output->setOutput();
        graph->dataMalloc();
        weight->copyin(vector<float>{1, 0, 0, 1});
    }

    void reshape(int batch) {
        if (input->getDims() == Shape{batch, 2})
            return;
        input->setShape({batch, 2});
        graph->shape_infer();
        graph->dataMalloc();
    }

    void run(const vector<float> &values) {
        IT_ASSERT(values.size() == input->size());
        input->copyin(values);
        runtime->runWithCudaGraph(graph);
    }

    bool outputEquals(const vector<float> &expected) const {
        return output->clone(NativeCpuRuntimeObj::getInstance())
            ->equalData(expected);
    }
};

vector<float> increasingValues(int batch, float offset = 0) {
    vector<float> values(batch * 2);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = static_cast<float>(i) + offset;
    return values;
}
} // namespace

TEST(TestCudaRuntime, CudaGraphCapturesFirstRunAndReplays) {
    auto runtime = make_ref<CudaRuntimeObj>();
    CudaGraphFixture fixture(runtime);

    auto values = increasingValues(8, -4);
    fixture.run(values);
    EXPECT_EQ(runtime->getCudaGraphCaptureCount(), 1u);
    EXPECT_EQ(runtime->getCudaGraphCacheSize(), 1u);

    vector<float> expected(values.size());
    std::transform(values.begin(), values.end(), expected.begin(),
                   [](float value) { return std::max(value, 0.0f); });
    EXPECT_TRUE(fixture.outputEquals(expected));

    values = increasingValues(8, 1);
    fixture.run(values);
    EXPECT_EQ(runtime->getCudaGraphCaptureCount(), 1u);
    EXPECT_TRUE(fixture.outputEquals(values));
}

TEST(TestCudaRuntime, CudaGraphReusesPreviousShapeInSameStorage) {
    auto runtime = make_ref<CudaRuntimeObj>();
    CudaGraphFixture fixture(runtime, 8);

    fixture.run(increasingValues(8));
    fixture.reshape(4);
    fixture.run(increasingValues(4));
    EXPECT_EQ(runtime->getCudaGraphCaptureCount(), 2u);

    fixture.reshape(8);
    auto values = increasingValues(8, 10);
    fixture.run(values);
    EXPECT_EQ(runtime->getCudaGraphCaptureCount(), 2u);
    EXPECT_EQ(runtime->getCudaGraphCacheSize(), 2u);
    EXPECT_TRUE(fixture.outputEquals(values));
}

TEST(TestCudaRuntime, CudaGraphIgnoresTensorContents) {
    auto runtime = make_ref<CudaRuntimeObj>();
    CudaGraphFixture fixture(runtime, 2);

    fixture.run(vector<float>{1, -2, 3, -4});
    fixture.weight->copyin(vector<float>{2, 0, 0, 3});
    fixture.run(vector<float>{-1, 2, 3, -4});

    EXPECT_EQ(runtime->getCudaGraphCaptureCount(), 1u);
    EXPECT_TRUE(fixture.outputEquals(vector<float>{0, 6, 6, 0}));
}

TEST(TestCudaRuntime, CudaGraphCapturesEluOnRuntimeStream) {
    auto runtime = make_ref<CudaRuntimeObj>();
    Graph graph = make_ref<GraphObj>(runtime);
    auto input = graph->addTensor({4}, DataType::Float32);
    input->setInput();
    auto output = graph->addOp<EluObj>(input, nullptr, 1.0f)->getOutput();
    output->setOutput();
    graph->dataMalloc();
    input->copyin(vector<float>{-1, 0, 1, 2});

    runtime->runWithCudaGraph(graph);
    runtime->runWithCudaGraph(graph);
    EXPECT_EQ(runtime->getCudaGraphCaptureCount(), 1u);
    auto cpuOutput = output->clone(NativeCpuRuntimeObj::getInstance());
    EXPECT_TRUE(cpuOutput->equalData(vector<float>{-0.63212055f, 0, 1, 2}));
}

TEST(TestCudaRuntime, CudaGraphInvalidatesReplacedStorage) {
    auto runtime = make_ref<CudaRuntimeObj>();
    CudaGraphFixture fixture(runtime, 8);

    fixture.run(increasingValues(8));
    EXPECT_EQ(runtime->getCudaGraphCacheSize(), 1u);

    fixture.reshape(1024);
    EXPECT_EQ(runtime->getCudaGraphCacheSize(), 0u);
    fixture.run(increasingValues(1024));
    EXPECT_EQ(runtime->getCudaGraphCaptureCount(), 2u);

    fixture.reshape(8);
    fixture.run(increasingValues(8));
    EXPECT_EQ(runtime->getCudaGraphCaptureCount(), 3u);
    fixture.graph->trimMemory();
    EXPECT_EQ(runtime->getCudaGraphCacheSize(), 0u);

    auto values = increasingValues(8, 3);
    fixture.run(values);
    EXPECT_EQ(runtime->getCudaGraphCaptureCount(), 4u);
    EXPECT_TRUE(fixture.outputEquals(values));
}

TEST(TestCudaRuntime, CudaGraphInvalidatesTopologyChanges) {
    auto runtime = make_ref<CudaRuntimeObj>();
    CudaGraphFixture fixture(runtime, 2);
    fixture.run(increasingValues(2));
    EXPECT_EQ(runtime->getCudaGraphCacheSize(), 1u);

    auto extraInput = fixture.graph->addTensor({1}, DataType::Float32);
    extraInput->setInput();
    EXPECT_EQ(runtime->getCudaGraphCacheSize(), 0u);
    fixture.graph->dataMalloc();
    extraInput->copyin(vector<float>{1});
    fixture.run(increasingValues(2, 2));

    EXPECT_EQ(runtime->getCudaGraphCaptureCount(), 2u);
    EXPECT_EQ(runtime->getCudaGraphCacheSize(), 1u);
    EXPECT_TRUE(fixture.outputEquals(increasingValues(2, 2)));
}

TEST(TestCudaRuntime, CudaGraphUsesBoundedLruCache) {
    auto runtime = make_ref<CudaRuntimeObj>(0, 2);
    CudaGraphFixture fixture(runtime, 8);

    fixture.run(increasingValues(8));
    fixture.reshape(6);
    fixture.run(increasingValues(6));
    fixture.reshape(8);
    fixture.run(increasingValues(8));
    EXPECT_EQ(runtime->getCudaGraphCaptureCount(), 2u);

    fixture.reshape(4);
    fixture.run(increasingValues(4));
    EXPECT_EQ(runtime->getCudaGraphCaptureCount(), 3u);
    EXPECT_EQ(runtime->getCudaGraphCacheSize(), 2u);

    fixture.reshape(8);
    fixture.run(increasingValues(8));
    EXPECT_EQ(runtime->getCudaGraphCaptureCount(), 3u);
    fixture.reshape(6);
    fixture.run(increasingValues(6));
    EXPECT_EQ(runtime->getCudaGraphCaptureCount(), 4u);
    EXPECT_EQ(runtime->getCudaGraphCacheSize(), 2u);
}

TEST(TestCudaRuntime, CudaGraphCachesMultipleGraphsPerRuntime) {
    auto runtime = make_ref<CudaRuntimeObj>();
    {
        CudaGraphFixture first(runtime, 2);
        CudaGraphFixture second(runtime, 3);
        first.run(increasingValues(2));
        second.run(increasingValues(3));
        first.run(increasingValues(2, 10));
        second.run(increasingValues(3, 20));

        EXPECT_EQ(runtime->getCudaGraphCaptureCount(), 2u);
        EXPECT_EQ(runtime->getCudaGraphCacheSize(), 2u);

        first.graph.reset();
        EXPECT_EQ(runtime->getCudaGraphCacheSize(), 1u);
        second.run(increasingValues(3, 30));
        EXPECT_EQ(runtime->getCudaGraphCaptureCount(), 2u);
    }
    EXPECT_EQ(runtime->getCudaGraphCacheSize(), 0u);
}

TEST(TestCudaRuntime, CudaGraphSerializesThreadsOnOneRuntime) {
    auto runtime = make_ref<CudaRuntimeObj>();
    CudaGraphFixture first(runtime, 2);
    CudaGraphFixture second(runtime, 3);
    first.input->copyin(increasingValues(2));
    second.input->copyin(increasingValues(3));
    std::atomic<bool> succeeded{true};

    auto worker = [&succeeded](CudaGraphFixture &fixture) {
        try {
            for (int i = 0; i < 20; ++i)
                fixture.runtime->runWithCudaGraph(fixture.graph);
        } catch (...) {
            succeeded = false;
        }
    };
    std::thread firstThread(worker, std::ref(first));
    std::thread secondThread(worker, std::ref(second));
    firstThread.join();
    secondThread.join();

    EXPECT_TRUE(succeeded);
    EXPECT_EQ(runtime->getCudaGraphCaptureCount(), 2u);
    EXPECT_EQ(runtime->getCudaGraphCacheSize(), 2u);
}

TEST(TestCudaRuntime, CudaGraphUsesIndependentRuntimeStreams) {
    auto secondRuntime = make_ref<CudaRuntimeObj>();
    CudaGraphFixture second(secondRuntime, 3);
    {
        auto firstRuntime = make_ref<CudaRuntimeObj>();
        CudaGraphFixture first(firstRuntime, 2);

        first.run(increasingValues(2));
        second.run(increasingValues(3));
        first.run(increasingValues(2, 10));
        second.run(increasingValues(3, 20));

        EXPECT_EQ(firstRuntime->getCudaGraphCaptureCount(), 1u);
        EXPECT_TRUE(first.outputEquals(increasingValues(2, 10)));
    }

    second.run(increasingValues(3, 30));
    EXPECT_EQ(secondRuntime->getCudaGraphCaptureCount(), 1u);
    EXPECT_TRUE(second.outputEquals(increasingValues(3, 30)));
}

TEST(TestCudaRuntime, CudaGraphRecoversAfterCaptureFailure) {
    auto runtime = make_ref<CudaRuntimeObj>();
    CudaGraphFixture valid(runtime, 2);
    valid.run(increasingValues(2));

    Graph invalid = make_ref<GraphObj>(runtime);
    auto input = invalid->addTensor({2, 2}, DataType::Float32);
    input->setInput();
    invalid->addOp<DropoutObj>(input, nullptr, nullptr, 0, false);
    invalid->dataMalloc();
    input->copyin(increasingValues(2));

    testing::internal::CaptureStderr();
    EXPECT_THROW(runtime->runWithCudaGraph(invalid), Exception);
    const auto cleanupLog = testing::internal::GetCapturedStderr();
    EXPECT_NE(cleanupLog.find("cudaStreamEndCapture after failure"),
              string::npos);
    EXPECT_EQ(runtime->getCudaGraphCaptureCount(), 1u);
    EXPECT_EQ(runtime->getCudaGraphCacheSize(), 1u);

    valid.run(increasingValues(2, 5));
    EXPECT_EQ(runtime->getCudaGraphCaptureCount(), 1u);
    EXPECT_TRUE(valid.outputEquals(increasingValues(2, 5)));
}

TEST(TestCudaRuntime, CudaGraphCacheCanBeCleared) {
    EXPECT_THROW(make_ref<CudaRuntimeObj>(0, 0), Exception);

    auto runtime = make_ref<CudaRuntimeObj>();
    CudaGraphFixture fixture(runtime, 2);
    fixture.run(increasingValues(2));
    runtime->clearCudaGraphCache();
    EXPECT_EQ(runtime->getCudaGraphCacheSize(), 0u);
    EXPECT_EQ(runtime->getCudaGraphCaptureCount(), 1u);

    fixture.run(increasingValues(2));
    EXPECT_EQ(runtime->getCudaGraphCaptureCount(), 2u);
}

} // namespace infini
