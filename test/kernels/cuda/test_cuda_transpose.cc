#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "operators/transpose.h"

#include "test.h"

namespace infini {

template <class T>
void testTranspose(
    const std::function<void(void *, size_t, DataType)> &generator,
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
    vector<int> permute = {0, 2, 1, 3};
    auto gpuOp = cudaGraph->addOp<T>(inputGpu, nullptr, permute);
    cudaGraph->dataMalloc();
    cudaRuntime->run(cudaGraph);
    auto outputGpu = gpuOp->getOutput();
    auto oCpu = outputGpu->clone(cpuRuntime);
    // Check
    // inputCpu->printData();
    // oCpu->printData();
    EXPECT_TRUE(oCpu->equalData(vector<float>{0, 1, 2,  3,  12, 13, 14, 15,
                                              4, 5, 6,  7,  16, 17, 18, 19,
                                              8, 9, 10, 11, 20, 21, 22, 23}));
}

TEST(cuda_Transpose, run) {
    testTranspose<TransposeObj>(IncrementalGenerator(), Shape{1, 2, 3, 4});
}

} // namespace infini
