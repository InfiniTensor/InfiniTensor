#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/global_pool.h"
#include "test.h"

namespace infini {
using ExpectOutput = vector<float>;

template <class T, typename std::enable_if<std::is_base_of<GlobalPoolObj, T>{},
                                           int>::type = 0>
void testGlobalPool(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &shape, const ExpectOutput &ansVec) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    // Build input data on CPU
    Tensor i0cpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);

    // Build CUDA graph
    Graph g = make_ref<GraphObj>(cudaRuntime);
    auto i0 = g->cloneTensor(i0cpu);
    auto pool = g->addOp<T>(i0, nullptr);

    // allocate CUDA memory
    g->dataMalloc();
    i0->setData(generator);

    // Execute on CUDA
    cudaRuntime->run(g);

    // clone CUDA output to CPU
    auto o0 = pool->getOutput();
    auto cpuo0 = o0->clone(cpuRuntime);

    // check results on CPU
    EXPECT_TRUE(cpuo0->equalData(ansVec));
}

TEST(CUDA_GlobalAvgPool, run) {
    testGlobalPool<GlobalAvgPoolObj>(IncrementalGenerator(), Shape{1, 2, 5, 5},
                                     ExpectOutput{12, 37});
}

} // namespace infini
