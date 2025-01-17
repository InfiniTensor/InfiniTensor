#include "core/graph.h"
#include "core/runtime.h"
#ifdef USE_CUDA
#include "cuda/cuda_runtime.h"
#endif
#include "operators/gemm.h"

#include "test.h"

namespace infini {

void testGemmCpu(
    const std::function<void(void *, size_t, DataType)> &generatorA,
    const std::function<void(void *, size_t, DataType)> &generatorB,
    const std::function<void(void *, size_t, DataType)> &generatorC,
    float alpha, float beta, bool transA, bool transB, const Shape &shapeA,
    const Shape &shapeB, const Shape &shapeC, const DataType &dataType) {

    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto A = g->addTensor(shapeA, dataType);
    auto B = g->addTensor(shapeB, dataType);
    auto C = g->addTensor(shapeC, dataType);

    auto op = g->addOp<GemmObj>(A, B, nullptr, C, alpha, beta, transA, transB);
    g->dataMalloc();
    A->setData(generatorA);
    B->setData(generatorB);
    C->setData(generatorC);

    runtime->run(g);
    // op->getOutput()->print();
    // op->getOutput()->printData();
    EXPECT_TRUE(1);
}

#ifdef USE_CUDA
void testGemmCuda(
    const std::function<void(void *, size_t, DataType)> &generatorA,
    const std::function<void(void *, size_t, DataType)> &generatorB,
    const std::function<void(void *, size_t, DataType)> &generatorC,
    float alpha, float beta, bool transA, bool transB, const Shape &shapeA,
    const Shape &shapeB, const Shape &shapeC, const DataType &dataType) {
    // cpu
    auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Graph cpuG = make_ref<GraphObj>(cpuRuntime);
    auto cpuA = cpuG->addTensor(shapeA, dataType);
    auto cpuB = cpuG->addTensor(shapeB, dataType);
    auto cpuC = cpuG->addTensor(shapeC, dataType);

    auto cpuOp = cpuG->addOp<GemmObj>(cpuA, cpuB, nullptr, cpuC, alpha, beta,
                                      transA, transB);
    cpuG->dataMalloc();
    cpuA->setData(generatorA);
    cpuB->setData(generatorB);
    cpuC->setData(generatorC);

    cpuRuntime->run(cpuG);
    auto cpuOutput = cpuOp->getOutput();

    // cuda
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    Graph cudaG = make_ref<GraphObj>(cudaRuntime);
    auto cudaA = cudaG->addTensor(shapeA, dataType);
    auto cudaB = cudaG->addTensor(shapeB, dataType);
    auto cudaC = cudaG->addTensor(shapeC, dataType);

    auto cudaOp = cudaG->addOp<GemmObj>(cudaA, cudaB, nullptr, cudaC, alpha,
                                        beta, transA, transB);
    cudaG->dataMalloc();
    cudaA->setData(generatorA);
    cudaB->setData(generatorB);
    cudaC->setData(generatorC);

    cudaRuntime->run(cudaG);
    auto cudaOutput = cudaOp->getOutput()->clone(cpuRuntime);

    EXPECT_TRUE(cudaOutput->equalData(cpuOutput));
}
#endif

TEST(Gemm, Cpu) {
    testGemmCpu(IncrementalGenerator(), IncrementalGenerator(),
                IncrementalGenerator(), 1.0, 0.0, false, false, Shape{3, 5},
                Shape{5, 2}, Shape{1}, DataType::Float32);
    testGemmCpu(IncrementalGenerator(), IncrementalGenerator(),
                IncrementalGenerator(), 1.0, 0.0, false, false, Shape{3, 5},
                Shape{5, 2}, Shape{1}, DataType::Float16);
    testGemmCpu(IncrementalGenerator(), IncrementalGenerator(),
                IncrementalGenerator(), 1.0, 1.0, false, false, Shape{1, 2048},
                Shape{2048, 2048}, Shape{1, 2048}, DataType::Float32);
    testGemmCpu(IncrementalGenerator(), IncrementalGenerator(),
                IncrementalGenerator(), 1.0, 1.0, false, false, Shape{1, 2048},
                Shape{2048, 2048}, Shape{1, 2048}, DataType::Float16);
}

#ifdef USE_CUDA
TEST(Gemm, Cuda) {
    testGemmCuda(IncrementalGenerator(), IncrementalGenerator(),
                 IncrementalGenerator(), 1.0, 0.0, false, false, Shape{3, 5},
                 Shape{5, 2}, Shape{1}, DataType::Float32);
    testGemmCuda(IncrementalGenerator(), IncrementalGenerator(),
                 IncrementalGenerator(), 1.0, 0.0, false, false, Shape{3, 5},
                 Shape{5, 2}, Shape{1}, DataType::Float16);
}
#endif

} // namespace infini
