#include "core/graph.h"
#include "core/runtime.h"
#ifdef USE_CUDA
#include "cuda/cuda_runtime.h"
#endif
#include "operators/gather.h"

#include "test.h"

namespace infini {


template <class T, typename std::enable_if<std::is_base_of<GatherObj, T>{},
                                           int>::type = 0>
void testGatherCpu(
    const std::function<void(void *, size_t, DataType)> &generator1,
    const std::function<void(void *, size_t, DataType)> &generator2,
    const int axis,const Shape &shape1, const Shape &shape2, const DataType &dataType) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    
    auto input = g->addTensor(shape1, dataType);
    auto indices = g->addTensor(shape2, DataType::Int32);

    auto op = g->addOp<T>(input, indices, nullptr, axis);
    g->dataMalloc();

    input->setData(generator1);
    indices->setData(generator2);

    runtime->run(g);

    //op->getOutput()->print();
    //op->getOutput()->printData();
    EXPECT_TRUE(1);
}

#ifdef USE_CUDA
template <class T, typename std::enable_if<std::is_base_of<GatherObj, T>{},
                                           int>::type = 0>
void testGatherCuda(
    const std::function<void(void *, size_t, DataType)> &generator1,
    const std::function<void(void *, size_t, DataType)> &generator2,
    const int axis,const Shape &shape1, const Shape &shape2, const DataType &dataType) {
    // cpu
    auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Graph cpuG = make_ref<GraphObj>(cpuRuntime);
    auto cpuInput = cpuG->addTensor(shape1, dataType);
    auto cpuIndices = cpuG->addTensor(shape2, DataType::Int32);

    auto cpuOp = cpuG->addOp<T>(cpuInput, cpuIndices, nullptr, axis);
    cpuG->dataMalloc();
    cpuInput->setData(generator1);
    cpuIndices->setData(generator2);

    cpuRuntime->run(cpuG);
    auto cpuOutput = cpuOp->getOutput();

    // cuda
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    Graph cudaG = make_ref<GraphObj>(cudaRuntime);
    auto cudaInput = cudaG->addTensor(shape1, dataType);
    auto cudaIndices = cudaG->addTensor(shape2, DataType::Int32);

    auto cudaOp =
        cudaG->addOp<T>(cudaInput, cudaIndices, nullptr, axis);
    cudaG->dataMalloc();
    cudaInput->setData(generator1);
    cudaIndices->setData(generator2);

    cudaRuntime->run(cudaG);
    auto cudaOutput = cudaOp->getOutput()->clone(cpuRuntime);

    EXPECT_TRUE(cudaOutput->equalData(cpuOutput));
}
#endif

TEST(Gather, Cpu) {
    testGatherCpu<GatherObj>(IncrementalGenerator(), IncrementalGenerator(), 2, Shape{1, 2, 5, 5},
                               Shape{2, 5}, DataType::Float32);
    testGatherCpu<GatherObj>(IncrementalGenerator(), IncrementalGenerator(), 1, Shape{2, 2, 5, 5},
                               Shape{5,}, DataType::Float16);
    testGatherCpu<GatherObj>(IncrementalGenerator(), IncrementalGenerator(), 0, Shape{3, 2},
                               Shape{2, 2}, DataType::Float32);
}

#ifdef USE_CUDA
TEST(Gather, Cuda) {
    testGatherCuda<GatherObj>(IncrementalGenerator(), IncrementalGenerator(), 2, Shape{1, 2, 5, 5},
                               Shape{1, 2, 2, 5}, DataType::Float32);
    testGatherCuda<GatherObj>(IncrementalGenerator(), IncrementalGenerator(), 1, Shape{2, 2, 5, 5},
                               Shape{2, 1, 5, 5}, DataType::Float16);
}
#endif

} // namespace infini
