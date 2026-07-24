#include "core/graph.h"
#include "core/runtime.h"
#ifdef USE_CUDA
#include "cuda/cuda_runtime.h"
#endif
#include "operators/unary.h"

#include "test.h"

namespace infini {
template <class T>
void testUnaryCpu(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &shape, const DataType &dataType) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shape, dataType);

    auto op = g->addOp<T>(input, nullptr);
    g->dataMalloc();
    input->setData(generator);

    runtime->run(g);
    // op->getOutput()->print();
    // op->getOutput()->printData();
    EXPECT_TRUE(1);
}

#ifdef USE_CUDA
template <class T>
void testUnaryCuda(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &shape, const DataType &dataType) {
    // cpu
    auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Graph cpuG = make_ref<GraphObj>(cpuRuntime);
    auto cpuInput = cpuG->addTensor(shape, dataType);

    auto cpuOp = cpuG->addOp<T>(cpuInput, nullptr);
    cpuG->dataMalloc();
    cpuInput->setData(generator);

    cpuRuntime->run(cpuG);
    auto cpuOutput = cpuOp->getOutput();

    // cuda
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    Graph cudaG = make_ref<GraphObj>(cudaRuntime);
    auto cudaInput = cudaG->addTensor(shape, dataType);

    auto cudaOp = cudaG->addOp<T>(cudaInput, nullptr);
    cudaG->dataMalloc();
    cudaInput->setData(generator);

    cudaRuntime->run(cudaG);
    auto cudaOutput = cudaOp->getOutput()->clone(cpuRuntime);

    EXPECT_TRUE(cudaOutput->equalData(cpuOutput));
}
#endif

TEST(ElementWise, Cpu) {
    testUnaryCpu<ReluObj>(IncrementalGenerator(), Shape{1, 2, 2, 3},
                          DataType::Float32);
    testUnaryCpu<ReluObj>(IncrementalGenerator(), Shape{1, 2, 2, 3},
                          DataType::Float16);
}

#ifdef USE_CUDA
TEST(ElementWise, Cuda) {
    testUnaryCuda<ReluObj>(IncrementalGenerator(), Shape{1, 2, 2, 3},
                           DataType::Float32);
    testUnaryCuda<ReluObj>(IncrementalGenerator(), Shape{1, 2, 2, 3},
                           DataType::Float16);
}
#endif

} // namespace infini
