#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "operators/transpose.h"

#include "test.h"

namespace infini {

void testTranspose(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &shape, const Shape &permute, vector<float> ans) {
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
    auto gpuOp = cudaGraph->addOp<TransposeObj>(inputGpu, nullptr, permute);
    cudaGraph->dataMalloc();
    cudaRuntime->run(cudaGraph);
    auto outputGpu = gpuOp->getOutput();
    auto oCpu = outputGpu->clone(cpuRuntime);
    EXPECT_TRUE(oCpu->equalData(ans));
}

TEST(cuda_Transpose, run_generic) {
    testTranspose(IncrementalGenerator(), {1, 2, 3, 4}, {0, 2, 1, 3},
                  {0,  1,  2,  3,  12, 13, 14, 15, 4,  5,  6,  7,
                   16, 17, 18, 19, 8,  9,  10, 11, 20, 21, 22, 23});
}

TEST(cuda_Transpose, run_fast_last_dim) {
    testTranspose(IncrementalGenerator(), {1, 2, 3, 4}, {0, 2, 3, 1},
                  {0, 12, 1, 13, 2, 14, 3, 15, 4,  16, 5,  17,
                   6, 18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 23});
}

} // namespace infini
