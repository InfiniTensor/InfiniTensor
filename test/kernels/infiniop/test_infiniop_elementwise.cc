#include "core/graph.h"
#include "core/runtime.h"
#ifdef USE_CUDA
#include "cuda/cuda_runtime.h"
#endif
#include "operators/element_wise.h"

#include "test.h"

namespace infini {

template <class T>
void testElementWiseCpu(
    const std::function<void(void *, size_t, DataType)> &generator1,
    const std::function<void(void *, size_t, DataType)> &generator2,
    const Shape &shape1, const Shape &shape2, const DataType &dataType) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto t1 = g->addTensor(shape1, dataType);
    auto t2 = g->addTensor(shape2, dataType);

    auto op = g->addOp<T>(t1, t2, nullptr);
    g->dataMalloc();
    t1->setData(generator1);
    t2->setData(generator2);

    runtime->run(g);
    // op->getOutput()->print();
    // op->getOutput()->printData();
    EXPECT_TRUE(1);
}

#ifdef USE_CUDA
template <class T>
void testElementWiseCuda(
    const std::function<void(void *, size_t, DataType)> &generator1,
    const std::function<void(void *, size_t, DataType)> &generator2,
    const Shape &shape1, const Shape &shape2, const DataType &dataType) {
    // cpu
    auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Graph cpuG = make_ref<GraphObj>(cpuRuntime);
    auto cpuT1 = cpuG->addTensor(shape1, dataType);
    auto cpuT2 = cpuG->addTensor(shape2, dataType);

    auto cpuOp = cpuG->addOp<T>(cpuT1, cpuT2, nullptr);
    cpuG->dataMalloc();
    cpuT1->setData(generator1);
    cpuT2->setData(generator2);

    cpuRuntime->run(cpuG);
    auto cpuOutput = cpuOp->getOutput();

    // cuda
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    Graph cudaG = make_ref<GraphObj>(cudaRuntime);
    auto cudaT1 = cudaG->addTensor(shape1, dataType);
    auto cudaT2 = cudaG->addTensor(shape2, dataType);

    auto cudaOp = cudaG->addOp<T>(cudaT1, cudaT2, nullptr);
    cudaG->dataMalloc();
    cudaT1->setData(generator1);
    cudaT2->setData(generator2);

    cudaRuntime->run(cudaG);
    auto cudaOutput = cudaOp->getOutput()->clone(cpuRuntime);

    EXPECT_TRUE(cudaOutput->equalData(cpuOutput));
}
#endif

TEST(ElementWise, Cpu) {
    testElementWiseCpu<AddObj>(IncrementalGenerator(), IncrementalGenerator(),
                               Shape{1, 2, 2, 3, 1}, Shape{2, 1, 1},
                               DataType::Float32);
    testElementWiseCpu<AddObj>(IncrementalGenerator(), IncrementalGenerator(),
                               Shape{1, 2, 2, 3, 1}, Shape{2, 1, 1},
                               DataType::Float16);
}

#ifdef USE_CUDA
TEST(ElementWise, Cuda) {
    testElementWiseCuda<AddObj>(IncrementalGenerator(), IncrementalGenerator(),
                                Shape{1, 2, 2, 3, 1}, Shape{2, 1, 1},
                                DataType::Float32);
    testElementWiseCuda<AddObj>(IncrementalGenerator(), IncrementalGenerator(),
                                Shape{1, 2, 2, 3, 1}, Shape{2, 1, 1},
                                DataType::Float16);
}
#endif

} // namespace infini
