#include "core/graph.h"
#include "core/runtime.h"
#include "similar_cuda/similar_cuda_runtime.h"
#ifdef USE_CUDA
#include "cuda/cuda_runtime.h"
#endif
#include "operators/element_wise.h"

#include "test.h"

namespace infini {

void testElementWiseCpu(
    const std::function<void(void *, size_t, DataType)> &generatorA,
    const std::function<void(void *, size_t, DataType)> &generatorB,
    const Shape &shapeA, const Shape &shapeB, const DataType &dataType) {

    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto A = g->addTensor(shapeA, dataType);
    auto B = g->addTensor(shapeB, dataType);

    auto op = g->addOp<AddObj>(A, B, nullptr);
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
void testElementWiseCuda(
    const std::function<void(void *, size_t, DataType)> &generatorA,
    const std::function<void(void *, size_t, DataType)> &generatorB,
    const Shape &shapeA, const Shape &shapeB, const DataType &dataType) {
    // cpu
    auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Graph cpuG = make_ref<GraphObj>(cpuRuntime);
    auto cpuA = cpuG->addTensor(shapeA, dataType);
    auto cpuB = cpuG->addTensor(shapeB, dataType);

    auto cpuOp = cpuG->addOp<AddObj>(cpuA, cpuB, nullptr);
    cpuG->dataMalloc();
    cpuA->setData(generatorA);
    cpuB->setData(generatorB);

    cpuRuntime->run(cpuG);
    auto cpuOutput = cpuOp->getOutput();

    // cuda
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    Graph cudaG = make_ref<GraphObj>(cudaRuntime);
    auto cudaA = cudaG->addTensor(shapeA, dataType);
    auto cudaB = cudaG->addTensor(shapeB, dataType);

    auto cudaOp = cudaG->addOp<AddObj>(cudaA, cudaB, nullptr);
    cudaG->dataMalloc();
    cudaA->setData(generatorA);
    cudaB->setData(generatorB);

    cudaRuntime->run(cudaG);
    auto cudaOutput = cudaOp->getOutput()->clone(cpuRuntime);

    EXPECT_TRUE(cudaOutput->equalData(cpuOutput));
}
#endif

void testAddSimilarCuda(
    Device device,
    const std::function<void(void *, size_t, DataType)> &generatorA,
    const std::function<void(void *, size_t, DataType)> &generatorB,
    const Shape &shapeA,
    const Shape &shapeB, const DataType &dataType) {
    // cpu
    auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Graph cpuG = make_ref<GraphObj>(cpuRuntime);
    auto cpuA = cpuG->addTensor(shapeA, dataType);
    auto cpuB = cpuG->addTensor(shapeB, dataType);

    auto cpuOp = cpuG->addOp<AddObj>(cpuA, cpuB, nullptr);
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

    auto Op = G->addOp<AddObj>(A, B, nullptr);
    G->dataMalloc();
    A->setData(generatorA);
    B->setData(generatorB);

    Runtime->run(G);
    auto Output = Op->getOutput()->clone(cpuRuntime);
    A->clone(cpuRuntime)->printData();
    B->clone(cpuRuntime)->printData();
    Output->printData();

    EXPECT_TRUE(Output->equalData(cpuOutput));
}

TEST(Add, Cpu) {
    testElementWiseCpu(IncrementalGenerator(), IncrementalGenerator(),
                       Shape{3, 5}, Shape{3, 5}, DataType::Float32);
}

#ifdef USE_ILUVATAR
TEST(Add, Iluvatar) {
    testAddSimilarCuda(Device::ILUVATAR, IncrementalGenerator(), IncrementalGenerator(), Shape{3, 5}, Shape{3, 5}, DataType::Float32);
    testAddSimilarCuda(Device::ILUVATAR, IncrementalGenerator(), IncrementalGenerator(), Shape{4, 5}, Shape{4, 5}, DataType::Float16);
}
#endif

#ifdef USE_MOORE
TEST(Add, Moore) {
    testAddSimilarCuda(Device::MOORE, IncrementalGenerator(), IncrementalGenerator(), Shape{3, 5}, Shape{3, 5}, DataType::Float32);
    testAddSimilarCuda(Device::MOORE, IncrementalGenerator(), IncrementalGenerator(), Shape{3, 5}, Shape{3, 5}, DataType::Float16);
}
#endif

#ifdef USE_METAX
TEST(Add, Metax) {
    testAddSimilarCuda(Device::METAX, IncrementalGenerator(), IncrementalGenerator(), Shape{3, 5}, Shape{3, 5}, DataType::Float32);
    testAddSimilarCuda(Device::METAX, IncrementalGenerator(), IncrementalGenerator(), Shape{3, 5}, Shape{3, 5}, DataType::Float16);
}
#endif


#ifdef USE_CUDA
TEST(Add, Cuda) {
    testElementWiseCuda(IncrementalGenerator(), IncrementalGenerator(),
                        Shape{3, 5}, Shape{3, 5}, DataType::Float32);
                        testElementWiseCuda(IncrementalGenerator(), IncrementalGenerator(),
                        Shape{3, 5}, Shape{3, 5}, DataType::Float16);
}
#endif

} // namespace infini