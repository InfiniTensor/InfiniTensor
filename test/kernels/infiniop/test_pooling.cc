#include "core/graph.h"
#include "core/runtime.h"
#ifdef USE_CUDA
#include "cuda/cuda_runtime.h"
#endif
#include "operators/pooling.h"

#include "test.h"

namespace infini {
using KDPS = vector<int>;

template <class T, typename std::enable_if<std::is_base_of<PoolingObj, T>{},
                                           int>::type = 0>
void testPoolingCpu(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &shape, const KDPS &kdps, const DataType &dataType) {
    EXPECT_TRUE(kdps.size() == 8);
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shape, dataType);

    auto op = g->addOp<T>(input, nullptr, kdps[0], kdps[1], kdps[2], kdps[3],
                          kdps[4], kdps[5], kdps[6], kdps[7], 0);
    g->dataMalloc();
    input->setData(generator);

    runtime->run(g);
    // op->getOutput()->print();
    // op->getOutput()->printData();
    EXPECT_TRUE(1);
}

#ifdef USE_CUDA
template <class T, typename std::enable_if<std::is_base_of<PoolingObj, T>{},
                                           int>::type = 0>
void testPoolingCuda(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &shape, const KDPS &kdps, const DataType &dataType) {
    // cpu
    auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Graph cpuG = make_ref<GraphObj>(cpuRuntime);
    auto cpuInput = cpuG->addTensor(shape, dataType);

    auto cpuOp = cpuG->addOp<T>(cpuInput, nullptr, kdps[0], kdps[1], kdps[2],
                                kdps[3], kdps[4], kdps[5], kdps[6], kdps[7], 0);
    cpuG->dataMalloc();
    cpuInput->setData(generator);

    cpuRuntime->run(cpuG);
    auto cpuOutput = cpuOp->getOutput();

    // cuda
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    Graph cudaG = make_ref<GraphObj>(cudaRuntime);
    auto cudaInput = cudaG->addTensor(shape, dataType);

    auto cudaOp =
        cudaG->addOp<T>(cudaInput, nullptr, kdps[0], kdps[1], kdps[2], kdps[3],
                        kdps[4], kdps[5], kdps[6], kdps[7], 0);
    cudaG->dataMalloc();
    cudaInput->setData(generator);

    cudaRuntime->run(cudaG);
    auto cudaOutput = cudaOp->getOutput()->clone(cpuRuntime);

    EXPECT_TRUE(cudaOutput->equalData(cpuOutput));
}
#endif

TEST(Pooling, Cpu) {
    testPoolingCpu<MaxPoolObj>(IncrementalGenerator(), Shape{1, 2, 5, 5},
                               KDPS{3, 3, 1, 1, 1, 1, 2, 2}, DataType::Float32);
    testPoolingCpu<MaxPoolObj>(IncrementalGenerator(), Shape{1, 2, 5, 5},
                               KDPS{3, 3, 1, 1, 1, 1, 2, 2}, DataType::Float16);
    testPoolingCpu<AvgPoolObj>(IncrementalGenerator(), Shape{1, 2, 5, 5},
                               KDPS{3, 3, 1, 1, 1, 1, 2, 2}, DataType::Float32);
    testPoolingCpu<AvgPoolObj>(IncrementalGenerator(), Shape{1, 2, 5, 5},
                               KDPS{3, 3, 1, 1, 1, 1, 2, 2}, DataType::Float16);
}

#ifdef USE_CUDA
TEST(Pooling, Cuda) {
    testPoolingCuda<MaxPoolObj>(IncrementalGenerator(), Shape{1, 2, 5, 5},
                                KDPS{3, 3, 1, 1, 1, 1, 2, 2},
                                DataType::Float32);
    testPoolingCuda<MaxPoolObj>(IncrementalGenerator(), Shape{1, 2, 5, 5},
                                KDPS{3, 3, 1, 1, 1, 1, 2, 2},
                                DataType::Float16);
    testPoolingCuda<AvgPoolObj>(IncrementalGenerator(), Shape{1, 2, 5, 5},
                                KDPS{3, 3, 1, 1, 1, 1, 2, 2},
                                DataType::Float32);
    testPoolingCuda<AvgPoolObj>(IncrementalGenerator(), Shape{1, 2, 5, 5},
                                KDPS{3, 3, 1, 1, 1, 1, 2, 2},
                                DataType::Float16);
}
#endif

} // namespace infini
