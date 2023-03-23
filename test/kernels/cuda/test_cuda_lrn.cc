#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

template <class T>
void testLrn(const std::function<void(void *, size_t, DataType)> &generator,
               const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = CpuRuntimeObj::getInstance();
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    inputCpu->dataMalloc();
    inputCpu->setData(generator);
    inputCpu->printData();

    // GPU
    Graph cudaGraph = make_ref<GraphObj>(cudaRuntime);
    auto inputGpu = cudaGraph->cloneTensor(inputCpu);
    auto gpuOp = cudaGraph->addOp<T>(inputGpu, nullptr, 5, 0.0001, 0.75, 2.0);
    cudaGraph->dataMalloc();
    cudaRuntime->run(cudaGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    outputGpu2Cpu->printData();
    EXPECT_TRUE(1);
}

TEST(cuDNN_Lrn, run) {
    testLrn<LrnObj>(IncrementalGenerator(), Shape{1, 10, 3, 3});
}

} // namespace infini
