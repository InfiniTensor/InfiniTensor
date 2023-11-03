#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

template <class T>
void testUnary(const std::function<void(void *, size_t, DataType)> &generator,
               const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);

    // GPU
    Graph cudaGraph = make_ref<GraphObj>(cudaRuntime);
    auto inputGpu = cudaGraph->cloneTensor(inputCpu);
    auto gpuOp = cudaGraph->addOp<T>(inputGpu, nullptr);
    cudaGraph->dataMalloc();
    inputGpu->setData(generator);
    cudaRuntime->run(cudaGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    // CPU
    Graph cpuGraph = make_ref<GraphObj>(cpuRuntime);
    auto cpuOp = cpuGraph->addOp<T>(inputCpu, nullptr);
    cpuGraph->addTensor(inputCpu);
    cpuGraph->dataMalloc();
    inputCpu->setData(generator);
    cpuRuntime->run(cpuGraph);
    auto outputCpu = cpuOp->getOutput();
    // Check
    EXPECT_TRUE(outputCpu->equalData(outputGpu2Cpu));
}

TEST(cuDNN_Unary, run) {
    testUnary<ReluObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<AbsObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<SigmoidObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<TanhObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<HardSigmoidObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<HardSwishObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<SqrtObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<NegObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<ErfObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    // more shapes
    testUnary<SqrtObj>(IncrementalGenerator(), Shape{13});
    testUnary<SqrtObj>(IncrementalGenerator(), Shape{4, 3});
    testUnary<SqrtObj>(IncrementalGenerator(), Shape{2, 3, 4, 5, 6});

    testUnary<GeluObj>(IncrementalGenerator(), Shape{1});
    testUnary<GeluObj>(IncrementalGenerator(), Shape{1, 2});
    testUnary<GeluObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
}

} // namespace infini
