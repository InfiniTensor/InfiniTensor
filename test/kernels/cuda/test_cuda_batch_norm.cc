#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/batch_norm.h"
#include "test.h"

namespace infini {

TEST(CUDA_BatchNorm, run) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    // Build cpu graph
    Graph gCpu = make_ref<GraphObj>(cpuRuntime);
    auto iCpu = gCpu->addTensor(Shape{1, 3, 2, 2}, DataType::Float32);
    auto meanCpu = gCpu->addTensor(Shape{3}, DataType::Float32);
    auto varCpu = gCpu->addTensor(Shape{3}, DataType::Float32);
    auto scaleCpu = gCpu->addTensor(Shape{3}, DataType::Float32);
    auto biasCpu = gCpu->addTensor(Shape{3}, DataType::Float32);

    // Build input data on CPU
    gCpu->dataMalloc();
    iCpu->setData(IncrementalGenerator());
    meanCpu->copyin(vector<float>{1, 6, 9});
    varCpu->copyin(vector<float>{4, 1, 9});
    scaleCpu->setData(OneGenerator());
    biasCpu->setData(ZeroGenerator());

    // Build CUDA graph
    Graph g = make_ref<GraphObj>(cudaRuntime);
    auto i = g->cloneTensor(iCpu);
    auto mean = g->cloneTensor(meanCpu);
    auto var = g->cloneTensor(varCpu);
    auto scale = g->cloneTensor(scaleCpu);
    auto bias = g->cloneTensor(biasCpu);
    auto op =
        g->addOp<BatchNormObj>(i, nullptr, mean, var, scale, bias, 0.9, 0);

    // allocate CUDA memory
    g->dataMalloc();

    // Execute on CUDA
    cudaRuntime->run(g);

    // clone CUDA output to CPU
    auto o = op->getOutput();
    auto ocpu = o->clone(cpuRuntime);

    // check results on CPU
    EXPECT_TRUE(ocpu->equalData(vector<float>{
        -0.5, 0, 0.5, 1, -2, -1, 0, 1, -0.333333, 0, 0.333333, 0.666667}));
}

TEST(CUDA_BatchNorm, preservesInputShapeDuringRunAndCapture) {
    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph graph = make_ref<GraphObj>(cudaRuntime);
    auto input = graph->addTensor(Shape{2, 3}, DataType::Float32);
    auto mean = graph->addTensor(Shape{3}, DataType::Float32);
    auto var = graph->addTensor(Shape{3}, DataType::Float32);
    auto scale = graph->addTensor(Shape{3}, DataType::Float32);
    auto bias = graph->addTensor(Shape{3}, DataType::Float32);
    auto batchNorm = graph->addOp<BatchNormObj>(input, nullptr, mean, var,
                                                scale, bias, 0.9, 0);
    graph->dataMalloc();

    input->copyin(vector<float>{0, 1, 2, 3, 4, 5});
    mean->copyin(vector<float>{0, 1, 2});
    var->copyin(vector<float>{1, 1, 1});
    scale->copyin(vector<float>{1, 1, 1});
    bias->copyin(vector<float>{0, 0, 0});
    const Shape originalShape{2, 3};

    cudaRuntime->run(graph);
    EXPECT_EQ(input->getDims(), originalShape);
    cudaRuntime->runWithCudaGraph(graph);
    EXPECT_EQ(input->getDims(), originalShape);
    cudaRuntime->runWithCudaGraph(graph);
    EXPECT_EQ(input->getDims(), originalShape);
    EXPECT_EQ(cudaRuntime->getCudaGraphCaptureCount(), 1u);

    auto output =
        batchNorm->getOutput()->clone(NativeCpuRuntimeObj::getInstance());
    EXPECT_TRUE(output->equalData(vector<float>{0, 0, 0, 3, 3, 3}));
}
} // namespace infini
