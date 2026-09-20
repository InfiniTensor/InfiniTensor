#include "core/graph.h"
#include "core/runtime.h"
#ifdef USE_CUDA
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "cuda/gather.h"
#include "operators/gather.h"
#endif
#include "operators/gather.h"
#include "operators/concat.h"
#include "operators/reshape.h"
#include "operators/unary.h"
#include "operators/unsqueeze.h"
#include "test.h"

namespace infini {

namespace {

void runShapeTensorChain(const Runtime &runtime, bool useCudaGraph = false) {
    auto graph = make_ref<GraphObj>(runtime);

    auto input = graph->addTensor({1, 2, 3}, DataType::Float32);
    input->setInput();
    auto batchIndex = graph->addTensor({1}, DataType::Int64);
    batchIndex->setInput();
    auto tail = graph->addTensor({1}, DataType::Int64);
    tail->setInput();

    auto shape = graph->addOp<ShapeObj>(input, nullptr)->getOutput();
    auto batch = graph->addOp<GatherObj>(shape, batchIndex, nullptr, 0)->getOutput();
    auto target = graph->addOp<ConcatObj>(TensorVec{batch, tail}, nullptr, 0)->getOutput();
    auto output = graph->addOp<ReshapeObj>(input, target, nullptr)->getOutput();
    output->setOutput();
    EXPECT_EQ(graph->getDynamicShapeOperators().size(), 3u);

    graph->dataMalloc();
    batchIndex->copyin<int64_t>({0});
    tail->copyin<int64_t>({6});

    const auto run = [&](int batchSize) {
        SCOPED_TRACE("batch=" + std::to_string(batchSize));
        input->setShape({batchSize, 2, 3});
        graph->prepareDynamicShapes();
        graph->shape_infer();
        graph->dataMalloc();
        std::vector<float> values(static_cast<size_t>(batchSize) * 6);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = static_cast<float>(i + batchSize);
        input->copyin(values);
#ifdef USE_CUDA
        if (useCudaGraph) {
            auto cudaRuntime = as<CudaRuntimeObj>(runtime);
            cudaRuntime->runWithCudaGraph(graph);
        } else {
            runtime->run(graph);
        }
#else
        (void)useCudaGraph;
        runtime->run(graph);
#endif
        EXPECT_EQ(output->getDims(), (Shape{batchSize, 6}));
        auto cpuOutput = output->clone(NativeCpuRuntimeObj::getInstance());
        EXPECT_TRUE(cpuOutput->equalData(values));
    };

    run(1);
    run(2);
    run(8);
    run(3);
    run(5);
    run(1);
}

TEST(DynamicShape, CpuActivationStorageReusesCapacityAfterShrink) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    auto graph = make_ref<GraphObj>(runtime);
    auto input = graph->addTensor({1, 2, 3}, DataType::Float32);
    input->setInput();
    auto output = graph->addOp<IdentityObj>(input, nullptr)->getOutput();
    output->setOutput();

    graph->dataMalloc();
    input->setShape({64, 2, 3});
    graph->shape_infer();
    graph->dataMalloc();
    const auto grownStorage = input->getDataBlob()->getStorageId();
    const auto grownBytes = input->getDataBlob()->getBytes();

    input->setShape({2, 2, 3});
    graph->shape_infer();
    graph->dataMalloc();
    EXPECT_EQ(input->getDataBlob()->getStorageId(), grownStorage);
    EXPECT_EQ(input->getDataBlob()->getBytes(), input->getBytes());

    input->setShape({64, 2, 3});
    graph->shape_infer();
    graph->dataMalloc();
    EXPECT_EQ(input->getDataBlob()->getStorageId(), grownStorage);
    EXPECT_EQ(input->getDataBlob()->getBytes(), grownBytes);
}

} // namespace

TEST(DynamicShape, ShapeTensorChainReplansAndPreservesValuesOnCpu) {
    runShapeTensorChain(NativeCpuRuntimeObj::getInstance());
}

#ifdef USE_CUDA
TEST(DynamicShape, ShapeTensorChainReplansAndPreservesValuesOnCuda) {
    runShapeTensorChain(make_ref<CudaRuntimeObj>(0, 8, 64ull << 20));
}

TEST(DynamicShape, CudaGatherSupportsRankEight) {
    auto runtime = make_ref<CudaRuntimeObj>(0, 4, 64ull << 20);
    auto graph = make_ref<GraphObj>(runtime);
    const Shape inputShape{1, 1, 1, 1, 1, 1, 2, 3};
    auto input = graph->addTensor(inputShape, DataType::Float32);
    auto index = graph->addTensor({2}, DataType::Int64);
    input->setInput();
    index->setInput();
    auto output = graph->addOp<GatherObj>(input, index, nullptr, 7)->getOutput();
    output->setOutput();
    graph->dataMalloc();
    input->copyin(vector<float>{0, 1, 2, 3, 4, 5});
    index->copyin(vector<int64_t>{2, 0});
    runtime->run(graph);
    auto cpuOutput = output->clone(NativeCpuRuntimeObj::getInstance());
    EXPECT_EQ(output->getDims(), (Shape{1, 1, 1, 1, 1, 1, 2, 2}));
    EXPECT_TRUE(cpuOutput->equalData(vector<float>{2, 0, 5, 3}));
}

TEST(DynamicShape, CudaGatherSupportsScalarIndex) {
    auto runtime = make_ref<CudaRuntimeObj>(0, 4, 64ull << 20);
    auto graph = make_ref<GraphObj>(runtime);
    auto input = graph->addTensor({2, 3}, DataType::Float32);
    auto index = graph->addTensor({}, DataType::Int64);
    input->setInput();
    index->setInput();
    auto output = graph->addOp<GatherObj>(input, index, nullptr, 0)->getOutput();
    output->setOutput();
    graph->dataMalloc();
    input->copyin(vector<float>{0, 1, 2, 3, 4, 5});
    index->copyin(vector<int64_t>{1});
    runtime->run(graph);
    auto cpuOutput = output->clone(NativeCpuRuntimeObj::getInstance());
    EXPECT_EQ(output->getDims(), (Shape{3}));
    EXPECT_TRUE(cpuOutput->equalData(vector<float>{3, 4, 5}));
}

TEST(DynamicShape, CudaGraphRecapturesAcrossDynamicShapes) {
    auto runtime = make_ref<CudaRuntimeObj>(0, 8, 64ull << 20);
    runShapeTensorChain(runtime, true);
    EXPECT_GE(runtime->getCudaGraphCaptureCount(), 5u);
    EXPECT_LE(runtime->getCudaGraphCacheSize(), 8u);
}
#endif

} // namespace infini
