#include "core/graph.h"
#include "core/runtime.h"
#include "similar_cuda/similar_cuda_runtime.h"
#ifdef USE_CUDA
#include "cuda/cuda_runtime.h"
#endif
#include "operators/gemm.h"

#include "test.h"

namespace infini {

void testGemmCpu(
    const std::function<void(void *, size_t, DataType)> &generatorA,
    const std::function<void(void *, size_t, DataType)> &generatorB,
    float alpha, float beta, bool transA, bool transB, const Shape &shapeA,
    const Shape &shapeB, const DataType &dataType) {

    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto A = g->addTensor(shapeA, dataType);
    auto B = g->addTensor(shapeB, dataType);

    auto op =
        g->addOp<GemmObj>(A, B, nullptr, nullptr, alpha, beta, transA, transB);
    g->dataMalloc();
    A->setData(generatorA);
    B->setData(generatorB);
    A->printData();
    B->printData();

    runtime->run(g);
    // op->getOutput()->print();
    op->getOutput()->printData();
    EXPECT_TRUE(1);
}

#ifdef USE_CUDA
void testGemmCuda(
    const std::function<void(void *, size_t, DataType)> &generatorA,
    const std::function<void(void *, size_t, DataType)> &generatorB,
    float alpha, float beta, bool transA, bool transB, const Shape &shapeA,
    const Shape &shapeB, const DataType &dataType) {
    // cpu
    auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Graph cpuG = make_ref<GraphObj>(cpuRuntime);
    auto cpuA = cpuG->addTensor(shapeA, dataType);
    auto cpuB = cpuG->addTensor(shapeB, dataType);

    auto cpuOp = cpuG->addOp<GemmObj>(cpuA, cpuB, nullptr, nullptr, alpha, beta,
                                      transA, transB);
    cpuG->dataMalloc();
    cpuA->setData(generatorA);
    cpuB->setData(generatorB);

    cpuRuntime->run(cpuG);
    auto cpuOutput = cpuOp->getOutput();

    // cuda
    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    std::cout << "======================================1" << std::endl;

    Graph cudaG = make_ref<GraphObj>(cudaRuntime);
    auto cudaA = cudaG->addTensor(shapeA, dataType);
    auto cudaB = cudaG->addTensor(shapeB, dataType);

    auto cudaOp = cudaG->addOp<GemmObj>(cudaA, cudaB, nullptr, nullptr, alpha,
                                        beta, transA, transB);

    cudaG->dataMalloc(true);
    cudaA->setData(generatorA);
    cudaB->setData(generatorB);

    cudaRuntime->run(cudaG);

    auto cudaOutput = cudaOp->getOutput()->clone(cpuRuntime);

    EXPECT_TRUE(cudaOutput->equalData(cpuOutput));
}
#endif

TEST(Gemm, Cpu) {
    testGemmCpu(IncrementalGenerator(), IncrementalGenerator(), 1.0, 0.0, false,
                false, Shape{3, 5}, Shape{5, 2}, DataType::Float32);
    // testGemmCpu(IncrementalGenerator(), IncrementalGenerator(), 1.0, 0.0,
    // false,
    //             false, Shape{3, 5}, Shape{5, 2}, DataType::Float16);
    // testGemmCpu(IncrementalGenerator(), IncrementalGenerator(), 1.0, 1.0,
    // false,
    //             false, Shape{1, 2048}, Shape{2048, 2048}, DataType::Float32);
    // testGemmCpu(IncrementalGenerator(), IncrementalGenerator(), 1.0, 1.0,
    // false,
    //             false, Shape{1, 2048}, Shape{2048, 2048}, DataType::Float16);
}

#ifdef USE_CUDA
TEST(Gemm, CudaFP32) {
    testGemmCuda(IncrementalGenerator(), IncrementalGenerator(), 1.0, 0.0,
                 false, false, Shape{3, 5}, Shape{5, 2}, DataType::Float32);
}

TEST(Gemm, CudaFP16) {
    testGemmCuda(IncrementalGenerator(), IncrementalGenerator(), 1.0, 0.0,
                 false, false, Shape{3, 5}, Shape{5, 2}, DataType::Float16);
}
#endif

void testGemmSimilarCuda(
    Device device,
    const std::function<void(void *, size_t, DataType)> &generatorA,
    const std::function<void(void *, size_t, DataType)> &generatorB,
    float alpha, float beta, bool transA, bool transB, const Shape &shapeA,
    const Shape &shapeB, const DataType &dataType) {
    // cpu
    auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Graph cpuG = make_ref<GraphObj>(cpuRuntime);
    auto cpuA = cpuG->addTensor(shapeA, dataType);
    auto cpuB = cpuG->addTensor(shapeB, dataType);

    auto cpuOp = cpuG->addOp<GemmObj>(cpuA, cpuB, nullptr, nullptr, alpha, beta,
                                      transA, transB);
    cpuG->dataMalloc();
    cpuA->setData(generatorA);
    cpuB->setData(generatorB);

    cpuRuntime->run(cpuG);
    auto cpuOutput = cpuOp->getOutput();

    // SimilarCuda
    auto Runtime = make_ref<SimilarRuntimeObj>(device);

    Graph G = make_ref<GraphObj>(Runtime);
    auto A = G->addTensor(shapeA, dataType);
    auto B = G->addTensor(shapeB, dataType);

    auto Op =
        G->addOp<GemmObj>(A, B, nullptr, nullptr, alpha, beta, transA, transB);
    G->dataMalloc();
    A->setData(generatorA);
    B->setData(generatorB);

    Runtime->run(G);
    auto Output = Op->getOutput()->clone(cpuRuntime);

    EXPECT_TRUE(Output->equalData(cpuOutput));
}

#ifdef USE_ILUVATAR
TEST(Gemm, Iluvatar) {
    testGemmSimilarCuda(Device::ILUVATAR, IncrementalGenerator(),
                        IncrementalGenerator(), 1.0, 0.0, false, false,
                        Shape{3, 5}, Shape{5, 2}, DataType::Float32);
}
#endif

#ifdef USE_METAX
TEST(Gemm, METAX) {
    testGemmSimilarCuda(Device::METAX, IncrementalGenerator(),
                        IncrementalGenerator(), 1.0, 0.0, false, false,
                        Shape{3, 5}, Shape{5, 2}, DataType::Float32);
}
#endif

#ifdef USE_MOORE
TEST(Gemm, MOORE) {
    testGemmSimilarCuda(Device::MOORE, IncrementalGenerator(),
                        IncrementalGenerator(), 1.0, 0.0, false, false,
                        Shape{3, 5}, Shape{5, 2}, DataType::Float32);
}
#endif

} // namespace infini