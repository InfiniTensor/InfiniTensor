#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/pooling.h"
#include "test.h"

namespace infini {
using KDPS = vector<int>;
using ExpectOutput = vector<float>;

template <class T>
void testPoolCudnn(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &shape, const KDPS &kdps, const ExpectOutput &ansVec) {
    EXPECT_TRUE(kdps.size() == 8);
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    // Build input data on CPU
    Tensor i0cpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    i0cpu->dataMalloc();
    i0cpu->setData(generator);

    // Build CUDA graph
    Graph g = make_ref<GraphObj>(cudaRuntime);
    auto i0 = g->cloneTensor(i0cpu);
    auto pool = g->addOp<T>(i0, nullptr, kdps[0], kdps[1], kdps[2], kdps[3],
                            kdps[4], kdps[5], kdps[6], kdps[7]);

    // allocate CUDA memory
    g->dataMalloc();

    // Execute on CUDA
    cudaRuntime->run(g);

    // clone CUDA output to CPU
    auto o0 = pool->getOutput();
    auto cpuo0 = o0->clone(cpuRuntime);

    // check results on CPU
    EXPECT_TRUE(cpuo0->equalData(ansVec));
}

TEST(cuDNN_MaxPool, run) {
    testPoolCudnn<MaxPoolObj>(IncrementalGenerator(), Shape{1, 2, 5, 5},
                              KDPS{3, 3, 1, 1, 1, 1, 2, 2},
                              ExpectOutput{6, 8, 9, 16, 18, 19, 21, 23, 24, 31,
                                           33, 34, 41, 43, 44, 46, 48, 49});
}

TEST(cuDNN_AvgPool, run) {
    testPoolCudnn<AvgPoolObj>(
        IncrementalGenerator(), Shape{1, 2, 5, 5}, KDPS{3, 3, 1, 1, 1, 1, 2, 2},
        ExpectOutput{1.333333, 3.0000, 2.666667, 7.0000, 12.0000, 9.0000,
                     8.0000, 13.0000, 9.333333, 12.44444, 19.666667, 13.777778,
                     23.666667, 37.0000, 25.666667, 19.111111, 29.666667,
                     20.444444});
}

} // namespace infini
