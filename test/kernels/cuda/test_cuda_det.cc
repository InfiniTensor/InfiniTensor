#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "operators/det.h"

#include "test.h"

namespace infini {

template <class T>
void testDet(const std::function<void(void *, size_t, DataType)> &generator,
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
    auto gpuOp = cudaGraph->addOp<T>(inputGpu, nullptr, "normal");
    cudaGraph->dataMalloc();
    cudaRuntime->run(cudaGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);

    // Check
    inputCpu->printData();
    outputGpu2Cpu->printData();
    EXPECT_TRUE(1);
}

TEST(cuda_Det, run) { testDet<DetObj>(IncrementalGenerator(), Shape{1, 2, 2}); }

} // namespace infini
