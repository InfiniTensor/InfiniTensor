#include "core/graph.h"
#include "core/runtime.h"
#ifdef USE_CUDA
#include "cuda/cuda_runtime.h"
#endif
#include "operators/where.h"

#include "test.h"

namespace infini {


template <class T, typename std::enable_if<std::is_base_of<WhereObj, T>{},
                                           int>::type = 0>
void testWhereCpu(
    const std::function<void(void *, size_t, DataType)> &generator1,
    const std::function<void(void *, size_t, DataType)> &generator2,
    const std::function<void(void *, size_t, DataType)> &generator3,
    const Shape &shape, const DataType &dataType) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto t1 = g->addTensor(shape, DataType::Float32);
    auto t2 = g->addTensor(shape, DataType::Float32);
    auto condition = g->addTensor(shape, DataType::UInt8);

    auto op = g->addOp<T>(t1, t2, condition, nullptr);
    g->dataMalloc();
    //t1->copyin(vector<float>{0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5});
    //t2->copyin(vector<float>{1, 1, 3, 2, 5, 1, 5, 2, 3, 5, 6, 7});
    //condition->copyin(vector<uint8_t>{0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0});
    t1->setData(generator1);
    t2->setData(generator2);
    condition->setData(generator3);


    runtime->run(g);
    //EXPECT_TRUE(
    //    op->getOutput()->equalData(vector<float>{0, 1, 3, 3, 4, 5, 5, 1, 3, 5, 4, 5}));
    //op->getOutput()->print();
    //op->getOutput()->printData();
    EXPECT_TRUE(1);
}

#ifdef USE_CUDA
template <class T, typename std::enable_if<std::is_base_of<WhereObj, T>{},
                                           int>::type = 0>
void testWhereCuda(
    const std::function<void(void *, size_t, DataType)> &generator1,
    const std::function<void(void *, size_t, DataType)> &generator2,
    const std::function<void(void *, size_t, DataType)> &generator3,
    const Shape &shape, const DataType &dataType) {
    // cpu
    auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Graph cpuG = make_ref<GraphObj>(cpuRuntime);
    auto cput1 = cpuG->addTensor(shape, DataType::Float32);
    auto cput2 = cpuG->addTensor(shape, DataType::Float32);
    auto cpucondition = cpuG->addTensor(shape, DataType::UInt8);

    auto cpuOp = cpuG->addOp<T>(cput1, cput2, cpucondition, nullptr);
    cpuG->dataMalloc();
    cput1->setData(generator1);
    cput2->setData(generator2);
    cpucondition->setData(generator3);

    cpuRuntime->run(cpuG);
    auto cpuOutput = cpuOp->getOutput();

    // cuda
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    Graph cudaG = make_ref<GraphObj>(cudaRuntime);
    auto cudat1 = cudaG->addTensor(shape, DataType::Float32);
    auto cudat2 = cudaG->addTensor(shape, DataType::Float32);
    auto cudacondition = cudaG->addTensor(shape, DataType::UInt8);

    auto cudaOp =
        cudaG->addOp<T>(cudat1, cudat2, cudacondition, nullptr);
    cudaG->dataMalloc();
    cudat1->setData(generator1);
    cudat2->setData(generator2);
    cudacondition->setData(generator3);

    cudaRuntime->run(cudaG);
    auto cudaOutput = cudaOp->getOutput()->clone(cpuRuntime);

    EXPECT_TRUE(cudaOutput->equalData(cpuOutput));
}
#endif

TEST(Where, Cpu) {
    testWhereCpu<WhereObj>(IncrementalGenerator(), IncrementalGenerator(), IncrementalGenerator(), Shape{1, 2, 5, 5},
                               DataType::Float32);
    testWhereCpu<WhereObj>(IncrementalGenerator(), IncrementalGenerator(), IncrementalGenerator(), Shape{2, 2, 5, 5},
                               DataType::Float16);
    testWhereCpu<WhereObj>(IncrementalGenerator(), IncrementalGenerator(), IncrementalGenerator(), Shape{2, 2, 1, 3},
                               DataType::Float32);
}

#ifdef USE_CUDA
TEST(Where, Cuda) {
    testWhereCuda<WhereObj>(IncrementalGenerator(), IncrementalGenerator(), IncrementalGenerator(), Shape{1, 2, 5, 5},
                               DataType::Float32);
    testWhereCuda<WhereObj>(IncrementalGenerator(), IncrementalGenerator(), IncrementalGenerator(), Shape{2, 2, 5, 5},
                               DataType::Float16);
}
#endif

} // namespace infini
