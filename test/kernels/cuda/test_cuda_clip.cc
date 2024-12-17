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
    Tensor inputMin =
        make_ref<TensorObj>(Shape{}, DataType::Float32, cpuRuntime);
    Tensor inputMax =
        make_ref<TensorObj>(Shape{}, DataType::Float32, cpuRuntime);

    // GPU
    Graph cudaGraph = make_ref<GraphObj>(cudaRuntime);
    auto inputGpu = cudaGraph->cloneTensor(inputCpu);
    auto inputMinGpu = cudaGraph->cloneTensor(inputMin);
    auto inputMaxGpu = cudaGraph->cloneTensor(inputMax);
    float min = 2.0;
    float max = 4.0;
    auto gpuOp =
        cudaGraph->addOp<T>(inputGpu, nullptr, inputMinGpu, inputMaxGpu);
    cudaGraph->dataMalloc();
    inputMinGpu->copyin(vector<float>{min});
    inputMaxGpu->copyin(vector<float>{max});
    inputGpu->setData(generator);
    cudaRuntime->run(cudaGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    // CPU
    Graph cpuGraph = make_ref<GraphObj>(cpuRuntime);
    auto cpuOp = cpuGraph->addOp<T>(inputCpu, nullptr, inputMin, inputMax);
    cpuGraph->addTensor(inputCpu);
    cpuGraph->addTensor(inputMin);
    cpuGraph->addTensor(inputMax);
    cpuGraph->dataMalloc();
    inputMin->copyin(vector<float>{min});
    inputMax->copyin(vector<float>{max});
    inputCpu->setData(generator);
    cpuRuntime->run(cpuGraph);
    auto outputCpu = cpuOp->getOutput();
    // Check
    EXPECT_TRUE(outputCpu->equalData(outputGpu2Cpu));
}

TEST(cuDNN_Unary, run) {
    testClip<ClipObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
}

} // namespace infini
