#include "core/graph.h"
#include "core/runtime.h"
#ifdef USE_CUDA
#include "cuda/cuda_runtime.h"
#endif
#include "operators/where.h"

#include "test.h"

namespace infini {

template <class T>
void testWhereCpu(
    const std::function<void(void *, size_t, DataType)> &generator1,
    const std::function<void(void *, size_t, DataType)> &generator2,
    const Shape &x_shape, const Shape &y_shape, const Shape &cond_shape,
    const DataType &dataType) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto x = g->addTensor(x_shape, dataType);
    auto y = g->addTensor(y_shape, dataType);
    auto cond = g->addTensor(cond_shape, DataType::UInt8);

    auto op = g->addOp<T>(x, y, cond, nullptr);
    g->dataMalloc();
    x->setData(generator1);
    y->setData(generator1);
    cond->setData(generator2);

    runtime->run(g);
    // op->getOutput()->print();
    // op->getOutput()->printData();
    EXPECT_TRUE(1);
}

#ifdef USE_CUDA
template <class T>
void testWhereCuda(
    const std::function<void(void *, size_t, DataType)> &generator1,
    const std::function<void(void *, size_t, DataType)> &generator2,
    const Shape &x_shape, const Shape &y_shape, const Shape &cond_shape,
    const DataType &dataType) {
    // cpu
    auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Graph cpuG = make_ref<GraphObj>(cpuRuntime);
    auto cpuX = cpuG->addTensor(x_shape, dataType);
    auto cpuY = cpuG->addTensor(y_shape, dataType);
    auto cpuCond = cpuG->addTensor(cond_shape, DataType::UInt8);

    auto cpuOp = cpuG->addOp<T>(cpuX, cpuY, cpuCond, nullptr);
    cpuG->dataMalloc();
    cpuX->setData(generator1);
    cpuY->setData(generator1);
    cpuCond->setData(generator2);

    cpuRuntime->run(cpuG);
    auto cpuOutput = cpuOp->getOutput();

    // cuda
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    Graph cudaG = make_ref<GraphObj>(cudaRuntime);
    auto cudaX = cudaG->addTensor(x_shape, dataType);
    auto cudaY = cudaG->addTensor(y_shape, dataType);
    auto cudaCond = cudaG->addTensor(cond_shape, DataType::UInt8);
    auto cudaOp = cudaG->addOp<T>(cudaX, cudaY, cudaCond, nullptr);
    cudaG->dataMalloc();
    cudaX->setData(generator1);
    cudaY->setData(generator1);
    cudaCond->setData(generator2);

    cudaRuntime->run(cudaG);
    auto cudaOutput = cudaOp->getOutput()->clone(cpuRuntime);

    EXPECT_TRUE(cudaOutput->equalData(cpuOutput));
}
#endif

TEST(Where, Cpu) {
    testWhereCpu<WhereObj>(IncrementalGenerator(), OneGenerator(), Shape{2, 2},
                           Shape{2, 2}, Shape{2, 2}, DataType::Float32);
    testWhereCpu<WhereObj>(IncrementalGenerator(), ZeroGenerator(), Shape{2, 2},
                           Shape{2, 2}, Shape{2, 2}, DataType::Float32);
    testWhereCpu<WhereObj>(IncrementalGenerator(), OneGenerator(), Shape{1},
                           Shape{2, 2}, Shape{2, 2}, DataType::Float32);
    testWhereCpu<WhereObj>(IncrementalGenerator(), ZeroGenerator(), Shape{1},
                           Shape{2, 2}, Shape{2, 2}, DataType::Float32);
    testWhereCpu<WhereObj>(IncrementalGenerator(), OneGenerator(), Shape{2, 2},
                           Shape{1}, Shape{2, 2}, DataType::Float32);
    testWhereCpu<WhereObj>(IncrementalGenerator(), ZeroGenerator(), Shape{2, 2},
                           Shape{1}, Shape{2, 2}, DataType::Float32);
    testWhereCpu<WhereObj>(IncrementalGenerator(), OneGenerator(), Shape{2, 2},
                           Shape{2, 2}, Shape{1}, DataType::Float32);
    testWhereCpu<WhereObj>(IncrementalGenerator(), ZeroGenerator(), Shape{2, 2},
                           Shape{2, 2}, Shape{1}, DataType::Float32);
    testWhereCpu<WhereObj>(IncrementalGenerator(), OneGenerator(), Shape{2, 2},
                           Shape{2, 2}, Shape{2, 2}, DataType::Float16);
    testWhereCpu<WhereObj>(IncrementalGenerator(), ZeroGenerator(), Shape{2, 2},
                           Shape{2, 2}, Shape{2, 2}, DataType::Float16);
    testWhereCpu<WhereObj>(IncrementalGenerator(), OneGenerator(), Shape{1},
                           Shape{2, 2}, Shape{2, 2}, DataType::Float16);
    testWhereCpu<WhereObj>(IncrementalGenerator(), ZeroGenerator(), Shape{1},
                           Shape{2, 2}, Shape{2, 2}, DataType::Float16);
    testWhereCpu<WhereObj>(IncrementalGenerator(), OneGenerator(), Shape{2, 2},
                           Shape{1}, Shape{2, 2}, DataType::Float16);
    testWhereCpu<WhereObj>(IncrementalGenerator(), ZeroGenerator(), Shape{2, 2},
                           Shape{1}, Shape{2, 2}, DataType::Float16);
    testWhereCpu<WhereObj>(IncrementalGenerator(), OneGenerator(), Shape{2, 2},
                           Shape{2, 2}, Shape{1}, DataType::Float16);
    testWhereCpu<WhereObj>(IncrementalGenerator(), ZeroGenerator(), Shape{2, 2},
                           Shape{2, 2}, Shape{1}, DataType::Float16);
}

#ifdef USE_CUDA
TEST(Where, Cuda) {
    testWhereCuda<WhereObj>(IncrementalGenerator(), OneGenerator(), Shape{2, 2},
                            Shape{2, 2}, Shape{2, 2}, DataType::Float32);
    testWhereCuda<WhereObj>(IncrementalGenerator(), ZeroGenerator(),
                            Shape{2, 2}, Shape{2, 2}, Shape{2, 2},
                            DataType::Float32);
    testWhereCuda<WhereObj>(IncrementalGenerator(), OneGenerator(), Shape{1},
                            Shape{2, 2}, Shape{2, 2}, DataType::Float32);
    testWhereCuda<WhereObj>(IncrementalGenerator(), ZeroGenerator(), Shape{1},
                            Shape{2, 2}, Shape{2, 2}, DataType::Float32);
    testWhereCuda<WhereObj>(IncrementalGenerator(), OneGenerator(), Shape{2, 2},
                            Shape{1}, Shape{2, 2}, DataType::Float32);
    testWhereCuda<WhereObj>(IncrementalGenerator(), ZeroGenerator(),
                            Shape{2, 2}, Shape{1}, Shape{2, 2},
                            DataType::Float32);
    testWhereCuda<WhereObj>(IncrementalGenerator(), OneGenerator(), Shape{2, 2},
                            Shape{2, 2}, Shape{1}, DataType::Float32);
    testWhereCuda<WhereObj>(IncrementalGenerator(), ZeroGenerator(),
                            Shape{2, 2}, Shape{2, 2}, Shape{1},
                            DataType::Float32);
    testWhereCuda<WhereObj>(IncrementalGenerator(), OneGenerator(), Shape{2, 2},
                            Shape{2, 2}, Shape{2, 2}, DataType::Float16);
    testWhereCuda<WhereObj>(IncrementalGenerator(), ZeroGenerator(),
                            Shape{2, 2}, Shape{2, 2}, Shape{2, 2},
                            DataType::Float16);
    testWhereCuda<WhereObj>(IncrementalGenerator(), OneGenerator(), Shape{1},
                            Shape{2, 2}, Shape{2, 2}, DataType::Float16);
    testWhereCuda<WhereObj>(IncrementalGenerator(), ZeroGenerator(), Shape{1},
                            Shape{2, 2}, Shape{2, 2}, DataType::Float16);
    testWhereCuda<WhereObj>(IncrementalGenerator(), OneGenerator(), Shape{2, 2},
                            Shape{1}, Shape{2, 2}, DataType::Float16);
    testWhereCuda<WhereObj>(IncrementalGenerator(), ZeroGenerator(),
                            Shape{2, 2}, Shape{1}, Shape{2, 2},
                            DataType::Float16);
    testWhereCuda<WhereObj>(IncrementalGenerator(), OneGenerator(), Shape{2, 2},
                            Shape{2, 2}, Shape{1}, DataType::Float16);
}
#endif

} // namespace infini
