#include "core/graph.h"
#include "core/runtime.h"
#ifdef USE_CUDA
#include "cuda/cuda_runtime.h"
#endif
#include "operators/reduce.h"

#include "test.h"

namespace infini {

template <class T, typename std::enable_if<std::is_base_of<ReduceBaseObj, T>{},
                                           int>::type = 0>
void testReduceCpu(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &shape, const optional<vector<int>> &axes, bool keepDims, const DataType &dataType) {

    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shape, dataType);

    auto op = g->addOp<T>(input, nullptr, axes, keepDims);
    g->dataMalloc();
    input->setData(generator);

    runtime->run(g);
    // op->getOutput()->print();
    // op->getOutput()->printData();
    EXPECT_TRUE(1);
}

#ifdef USE_CUDA
template <class T, typename std::enable_if<std::is_base_of<ReduceBaseObj, T>{},
                                           int>::type = 0>
void testReduceCuda(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &shape, const optional<vector<int>> &axes, bool keepDims, const DataType &dataType){
    // cpu
    auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Graph cpuG = make_ref<GraphObj>(cpuRuntime);
    auto cpuInput = cpuG->addTensor(shape, dataType);

    auto cpuOp = cpuG->addOp<T>(cpuInput, nullptr, axes, keepDims);
    cpuG->dataMalloc();
    cpuInput->setData(generator);

    cpuRuntime->run(cpuG);
    auto cpuOutput = cpuOp->getOutput();

    // cuda
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    Graph cudaG = make_ref<GraphObj>(cudaRuntime);
    auto cudaInput = cudaG->addTensor(shape, dataType);

    auto cudaOp =
        cudaG->addOp<T>(cudaInput, nullptr, axes, keepDims);
    cudaG->dataMalloc();
    cudaInput->setData(generator);

    cudaRuntime->run(cudaG);
    auto cudaOutput = cudaOp->getOutput()->clone(cpuRuntime);

    EXPECT_TRUE(cudaOutput->equalData(cpuOutput));
}
#endif

TEST(Reduce, Cpu) {
    testReduceCpu<ReduceMaxObj>(IncrementalGenerator(), Shape{1, 2, 5, 5},
                                std::vector<int>{1,2}, 1, DataType::Float32);
    testReduceCpu<ReduceMaxObj>(IncrementalGenerator(), Shape{1, 3, 5, 5},
                                std::vector<int>{0, 2, 3}, 0, DataType::Float16);
    testReduceCpu<ReduceMinObj>(IncrementalGenerator(), Shape{4, 2, 5, 5},
                                std::vector<int>{0}, 1, DataType::Float32);
    testReduceCpu<ReduceMinObj>(IncrementalGenerator(), Shape{3, 2, 5, 5},
                                std::vector<int>{2}, 0, DataType::Float16);
    testReduceCpu<ReduceMeanObj>(IncrementalGenerator(), Shape{1, 2, 5, 5},
                                std::vector<int>{1,3}, 1, DataType::Float32);
    testReduceCpu<ReduceMeanObj>(IncrementalGenerator(), Shape{1, 8, 5, 5},
                                std::vector<int>{2}, 0, DataType::Float16);
    testReduceCpu<ReduceSumObj>(IncrementalGenerator(), Shape{7, 2, 5, 5},
                                std::vector<int>{1,3}, 1, DataType::Float32);
    testReduceCpu<ReduceSumObj>(IncrementalGenerator(), Shape{1, 2, 5, 5},
                                std::vector<int>{2}, 0, DataType::Float16);
}

#ifdef USE_CUDA
TEST(Reduce, Cuda) {
    testReduceCuda<ReduceMaxObj>(IncrementalGenerator(), Shape{1, 2, 5, 5},
                                std::vector<int>{1}, 1, DataType::Float32);
    testReduceCuda<ReduceMaxObj>(IncrementalGenerator(), Shape{1, 2, 5, 5},
                                std::vector<int>{2}, 0, DataType::Float16);
    testReduceCuda<ReduceMinObj>(IncrementalGenerator(), Shape{1, 2, 5, 5},
                                std::vector<int>{1}, 1, DataType::Float32);
    testReduceCuda<ReduceMinObj>(IncrementalGenerator(), Shape{1, 2, 5, 5},
                                std::vector<int>{2}, 0, DataType::Float16);
    testReduceCuda<ReduceMeanObj>(IncrementalGenerator(), Shape{1, 2, 5, 5},
                                std::vector<int>{1}, 1, DataType::Float32);
    testReduceCuda<ReduceMeanObj>(IncrementalGenerator(), Shape{1, 2, 5, 5},
                                std::vector<int>{2}, 0, DataType::Float16);
    testReduceCuda<ReduceSumObj>(IncrementalGenerator(), Shape{1, 2, 5, 5},
                                std::vector<int>{1}, 1, DataType::Float32);
    testReduceCuda<ReduceSumObj>(IncrementalGenerator(), Shape{1, 2, 5, 5},
                                std::vector<int>{2}, 0, DataType::Float16);
}
#endif

} // namespace infini
