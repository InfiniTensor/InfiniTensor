#include "core/graph.h"
#include "core/infini_runtime.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/dropout.h"
#include "operators/rms_norm.h"

#include "test.h"
#include <atomic>
#include <cmath>
#include <thread>

namespace infini {

namespace {
Ref<InfiniRuntimeObj> makeRuntime(size_t graphCacheCapacity = 16) {
    const auto deviceType = ::infini::rt::runtime_device_type();
    return make_ref<InfiniRuntimeObj>(
        string(::infini::rt::Device::StringFromType(deviceType)), 0,
        graphCacheCapacity);
}

class CaptureUnsupportedKernel : public Kernel {
    void compute(const Operator &op, const PerfRecord &,
                 const RuntimeObj *context) const override {
        compute(op, context);
    }

    void compute(const Operator &, const RuntimeObj *context) const override {
        auto runtime = dynamic_cast<const InfiniRuntimeObj *>(context);
        IT_ASSERT(runtime != nullptr);
        runtime->sync();
    }

    PerfRecord tune(const Operator &, const RuntimeObj *) const override {
        return make_ref<PerfRecordObj>();
    }
};

[[maybe_unused]] const bool captureUnsupportedKernelRegistered =
    KernelRegistry::getInstance().registerKernel(
        KernelAttrs{ExecutionProvider::Infini, OpType::Dropout},
        new CaptureUnsupportedKernel(), "CaptureUnsupportedKernel");

class GraphCaptureFixture {
  public:
    Ref<InfiniRuntimeObj> runtime;
    Graph graph;
    Tensor input;
    Tensor weight;
    Tensor output;

    explicit GraphCaptureFixture(Ref<InfiniRuntimeObj> runtime, int batch = 8)
        : runtime(std::move(runtime)),
          graph(make_ref<GraphObj>(this->runtime)) {
        input = graph->addTensor({batch, 2}, DataType::Float32);
        weight = graph->addTensor({2}, DataType::Float32);
        input->setInput();
        weight->setWeight();
        output = graph->addOp<RMSNormObj>(input, weight, nullptr)->getOutput();
        output->setOutput();
        graph->dataMalloc();
        weight->copyin(vector<float>{1, 1});
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
        runtime->runWithGraph(graph);
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

vector<float> rmsNormValues(const vector<float> &values, float firstWeight = 1,
                            float secondWeight = 1) {
    IT_ASSERT(values.size() % 2 == 0);
    vector<float> expected(values.size());
    for (size_t i = 0; i < values.size(); i += 2) {
        const auto first = values[i];
        const auto second = values[i + 1];
        const auto inverseRms =
            1.0f / std::sqrt((first * first + second * second) / 2.0f + 1e-6f);
        expected[i] = first * inverseRms * firstWeight;
        expected[i + 1] = second * inverseRms * secondWeight;
    }
    return expected;
}

struct GraphCaptureSupport {
    bool available;
    string reason;
};

const GraphCaptureSupport &graphCaptureSupport() {
    static const auto support = []() {
        try {
            auto runtime = makeRuntime();
            GraphCaptureFixture fixture(runtime, 2);
            fixture.run(increasingValues(2));
            return GraphCaptureSupport{true, {}};
        } catch (const std::exception &error) {
            return GraphCaptureSupport{false, error.what()};
        }
    }();
    return support;
}

class InfiniGraphCaptureTest : public ::testing::Test {
  protected:
    void SetUp() override {
        const auto &support = graphCaptureSupport();
        if (!support.available) {
            GTEST_SKIP() << "The installed InfiniRT/InfiniOps backend cannot "
                            "capture the test operator: "
                         << support.reason;
        }
    }
};
} // namespace

TEST_F(InfiniGraphCaptureTest, GraphCaptureCapturesFirstRunAndReplays) {
    auto runtime = makeRuntime();
    GraphCaptureFixture fixture(runtime);

    auto values = increasingValues(8, -4);
    fixture.run(values);
    EXPECT_EQ(runtime->getGraphCaptureCount(), 1u);
    EXPECT_EQ(runtime->getGraphCacheSize(), 1u);

    EXPECT_TRUE(fixture.outputEquals(rmsNormValues(values)));

    values = increasingValues(8, 1);
    fixture.run(values);
    EXPECT_EQ(runtime->getGraphCaptureCount(), 1u);
    EXPECT_TRUE(fixture.outputEquals(rmsNormValues(values)));
}

TEST_F(InfiniGraphCaptureTest, GraphCaptureReusesPreviousShapeInSameStorage) {
    auto runtime = makeRuntime();
    GraphCaptureFixture fixture(runtime, 8);

    fixture.run(increasingValues(8));
    fixture.reshape(4);
    fixture.run(increasingValues(4));
    EXPECT_EQ(runtime->getGraphCaptureCount(), 2u);

    fixture.reshape(8);
    auto values = increasingValues(8, 10);
    fixture.run(values);
    EXPECT_EQ(runtime->getGraphCaptureCount(), 2u);
    EXPECT_EQ(runtime->getGraphCacheSize(), 2u);
    EXPECT_TRUE(fixture.outputEquals(rmsNormValues(values)));
}

TEST_F(InfiniGraphCaptureTest, GraphCaptureIgnoresTensorContents) {
    auto runtime = makeRuntime();
    GraphCaptureFixture fixture(runtime, 2);

    fixture.run(vector<float>{1, -2, 3, -4});
    fixture.weight->copyin(vector<float>{2, 3});
    auto values = vector<float>{-1, 2, 3, -4};
    fixture.run(values);

    EXPECT_EQ(runtime->getGraphCaptureCount(), 1u);
    EXPECT_TRUE(fixture.outputEquals(rmsNormValues(values, 2, 3)));
}

TEST_F(InfiniGraphCaptureTest,
       GraphCaptureCapturesNativeKernelOnRuntimeStream) {
    auto runtime = makeRuntime();
    GraphCaptureFixture fixture(runtime, 2);
    const auto values = vector<float>{-1, 0, 1, 2};

    fixture.run(values);
    fixture.run(values);
    EXPECT_EQ(runtime->getGraphCaptureCount(), 1u);
    EXPECT_TRUE(fixture.outputEquals(rmsNormValues(values)));
}

TEST_F(InfiniGraphCaptureTest, GraphCaptureInvalidatesReplacedStorage) {
    auto runtime = makeRuntime();
    GraphCaptureFixture fixture(runtime, 8);

    fixture.run(increasingValues(8));
    EXPECT_EQ(runtime->getGraphCacheSize(), 1u);

    fixture.reshape(1024);
    EXPECT_EQ(runtime->getGraphCacheSize(), 0u);
    fixture.run(increasingValues(1024));
    EXPECT_EQ(runtime->getGraphCaptureCount(), 2u);

    fixture.reshape(8);
    fixture.run(increasingValues(8));
    EXPECT_EQ(runtime->getGraphCaptureCount(), 3u);
    fixture.graph->trimMemory();
    EXPECT_EQ(runtime->getGraphCacheSize(), 0u);

    auto values = increasingValues(8, 3);
    fixture.run(values);
    EXPECT_EQ(runtime->getGraphCaptureCount(), 4u);
    EXPECT_TRUE(fixture.outputEquals(rmsNormValues(values)));
}

TEST_F(InfiniGraphCaptureTest, GraphCaptureInvalidatesTopologyChanges) {
    auto runtime = makeRuntime();
    GraphCaptureFixture fixture(runtime, 2);
    fixture.run(increasingValues(2));
    EXPECT_EQ(runtime->getGraphCacheSize(), 1u);

    auto extraInput = fixture.graph->addTensor({1}, DataType::Float32);
    extraInput->setInput();
    EXPECT_EQ(runtime->getGraphCacheSize(), 0u);
    fixture.graph->dataMalloc();
    extraInput->copyin(vector<float>{1});
    fixture.run(increasingValues(2, 2));

    EXPECT_EQ(runtime->getGraphCaptureCount(), 2u);
    EXPECT_EQ(runtime->getGraphCacheSize(), 1u);
    EXPECT_TRUE(fixture.outputEquals(rmsNormValues(increasingValues(2, 2))));
}

TEST_F(InfiniGraphCaptureTest, GraphCaptureUsesBoundedLruCache) {
    auto runtime = makeRuntime(2);
    GraphCaptureFixture fixture(runtime, 8);

    fixture.run(increasingValues(8));
    fixture.reshape(6);
    fixture.run(increasingValues(6));
    fixture.reshape(8);
    fixture.run(increasingValues(8));
    EXPECT_EQ(runtime->getGraphCaptureCount(), 2u);

    fixture.reshape(4);
    fixture.run(increasingValues(4));
    EXPECT_EQ(runtime->getGraphCaptureCount(), 3u);
    EXPECT_EQ(runtime->getGraphCacheSize(), 2u);

    fixture.reshape(8);
    fixture.run(increasingValues(8));
    EXPECT_EQ(runtime->getGraphCaptureCount(), 3u);
    fixture.reshape(6);
    fixture.run(increasingValues(6));
    EXPECT_EQ(runtime->getGraphCaptureCount(), 4u);
    EXPECT_EQ(runtime->getGraphCacheSize(), 2u);
}

TEST_F(InfiniGraphCaptureTest, GraphCaptureCachesMultipleGraphsPerRuntime) {
    auto runtime = makeRuntime();
    {
        GraphCaptureFixture first(runtime, 2);
        GraphCaptureFixture second(runtime, 3);
        first.run(increasingValues(2));
        second.run(increasingValues(3));
        first.run(increasingValues(2, 10));
        second.run(increasingValues(3, 20));

        EXPECT_EQ(runtime->getGraphCaptureCount(), 2u);
        EXPECT_EQ(runtime->getGraphCacheSize(), 2u);

        first.graph.reset();
        EXPECT_EQ(runtime->getGraphCacheSize(), 1u);
        second.run(increasingValues(3, 30));
        EXPECT_EQ(runtime->getGraphCaptureCount(), 2u);
    }
    EXPECT_EQ(runtime->getGraphCacheSize(), 0u);
}

TEST_F(InfiniGraphCaptureTest, GraphCaptureSerializesThreadsOnOneRuntime) {
    auto runtime = makeRuntime();
    GraphCaptureFixture first(runtime, 2);
    GraphCaptureFixture second(runtime, 3);
    first.input->copyin(increasingValues(2));
    second.input->copyin(increasingValues(3));
    std::atomic<bool> succeeded{true};

    auto worker = [&succeeded](GraphCaptureFixture &fixture) {
        try {
            for (int i = 0; i < 20; ++i)
                fixture.runtime->runWithGraph(fixture.graph);
        } catch (...) {
            succeeded = false;
        }
    };
    std::thread firstThread(worker, std::ref(first));
    std::thread secondThread(worker, std::ref(second));
    firstThread.join();
    secondThread.join();

    EXPECT_TRUE(succeeded);
    EXPECT_EQ(runtime->getGraphCaptureCount(), 2u);
    EXPECT_EQ(runtime->getGraphCacheSize(), 2u);
}

TEST_F(InfiniGraphCaptureTest, GraphCaptureUsesIndependentRuntimeStreams) {
    auto secondRuntime = makeRuntime();
    GraphCaptureFixture second(secondRuntime, 3);
    {
        auto firstRuntime = makeRuntime();
        GraphCaptureFixture first(firstRuntime, 2);

        first.run(increasingValues(2));
        second.run(increasingValues(3));
        first.run(increasingValues(2, 10));
        second.run(increasingValues(3, 20));

        EXPECT_EQ(firstRuntime->getGraphCaptureCount(), 1u);
        EXPECT_TRUE(first.outputEquals(rmsNormValues(increasingValues(2, 10))));
    }

    second.run(increasingValues(3, 30));
    EXPECT_EQ(secondRuntime->getGraphCaptureCount(), 1u);
    EXPECT_TRUE(second.outputEquals(rmsNormValues(increasingValues(3, 30))));
}

TEST_F(InfiniGraphCaptureTest, GraphCaptureRecoversAfterCaptureFailure) {
    auto runtime = makeRuntime();
    GraphCaptureFixture valid(runtime, 2);
    valid.run(increasingValues(2));

    Graph invalid = make_ref<GraphObj>(runtime);
    auto input = invalid->addTensor({2, 2}, DataType::Float32);
    input->setInput();
    invalid->addOp<DropoutObj>(input, nullptr, nullptr, 0, false);
    invalid->dataMalloc();
    input->copyin(increasingValues(2));

    testing::internal::CaptureStderr();
    EXPECT_THROW(runtime->runWithGraph(invalid), Exception);
    const auto cleanupLog = testing::internal::GetCapturedStderr();
    EXPECT_NE(cleanupLog.find("InfiniRT StreamEndCapture after failure"),
              string::npos);
    EXPECT_EQ(runtime->getGraphCaptureCount(), 1u);
    EXPECT_EQ(runtime->getGraphCacheSize(), 1u);

    valid.run(increasingValues(2, 5));
    EXPECT_EQ(runtime->getGraphCaptureCount(), 1u);
    EXPECT_TRUE(valid.outputEquals(rmsNormValues(increasingValues(2, 5))));
}

TEST_F(InfiniGraphCaptureTest, GraphCaptureCacheCanBeCleared) {
    EXPECT_THROW(makeRuntime(0), Exception);

    auto runtime = makeRuntime();
    GraphCaptureFixture fixture(runtime, 2);
    fixture.run(increasingValues(2));
    runtime->clearGraphCache();
    EXPECT_EQ(runtime->getGraphCacheSize(), 0u);
    EXPECT_EQ(runtime->getGraphCaptureCount(), 1u);

    fixture.run(increasingValues(2));
    EXPECT_EQ(runtime->getGraphCaptureCount(), 2u);
}

} // namespace infini
