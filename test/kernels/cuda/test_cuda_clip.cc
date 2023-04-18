#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

template <class T>
void testClip(const std::function<void(void *, size_t, DataType)> &generator,
              const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    inputCpu->dataMalloc();
    inputCpu->setData(generator);

    // GPU
    Graph cudaGraph = make_ref<GraphObj>(cudaRuntime);
    auto inputGpu = cudaGraph->cloneTensor(inputCpu);
    float min = 2.0;
    float max = 4.0;
    auto gpuOp = cudaGraph->addOp<T>(inputGpu, nullptr, min, max);
    cudaGraph->dataMalloc();
    cudaRuntime->run(cudaGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    // CPU
    Graph cpuGraph = make_ref<GraphObj>(cpuRuntime);
    auto cpuOp = cpuGraph->addOp<T>(inputCpu, nullptr, min, max);
    cpuGraph->dataMalloc();
    cpuRuntime->run(cpuGraph);
    auto outputCpu = cpuOp->getOutput();
    // Check
    EXPECT_TRUE(outputCpu->equalData(outputGpu2Cpu));
}

TEST(cuDNN_Unary, run) {
    testClip<ClipObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
}

} // namespace infini
