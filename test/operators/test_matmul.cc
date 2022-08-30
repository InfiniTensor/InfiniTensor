
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/matmul.h"

#include "test.h"

namespace infini {
using ExpectOutput = vector<float>;

TEST(Matmul, ShapeInference) {
    auto runtime = CpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(runtime);
        auto A = g->addTensor(Shape{1, 3, 5});
        auto B = g->addTensor(Shape{1, 5, 2});
        auto matmul = g->addOp<MatmulObj>(A, B, nullptr);
        auto C = matmul->getOutputs()[0];
        EXPECT_EQ(C->getDims(), (Shape{1, 3, 2}));
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        auto A = g->addTensor(Shape{3, 5, 4});
        auto B = g->addTensor(Shape{3, 5, 2});
        auto matmul = g->addOp<MatmulObj>(A, B, nullptr, true, false);
        auto C = matmul->getOutputs()[0];
        EXPECT_EQ(C->getDims(), (Shape{3, 4, 2}));
    }
}
void testMatmulCuda(
    const std::function<void(void *, size_t, DataType)> &generatorA,
    const std::function<void(void *, size_t, DataType)> &generatorB,
    bool transA, bool transB, const Shape &shapeA, const Shape &shapeB,
    const ExpectOutput &ansVec) {
    auto cpuRuntime = CpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(cpuRuntime);
    auto ACpu = gCpu->addTensor(shapeA, DataType::Float32);
    auto BCpu = gCpu->addTensor(shapeB, DataType::Float32);
    gCpu->dataMalloc();
    ACpu->setData(generatorA);
    BCpu->setData(generatorB);

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    auto gCuda = make_ref<GraphObj>(cudaRuntime);
    auto ACuda = gCuda->cloneTensor(ACpu);
    auto BCuda = gCuda->cloneTensor(BCpu);
    auto matmul =
        gCuda->addOp<MatmulObj>(ACuda, BCuda, nullptr, transA, transB);

    // allocate CUDA memory
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    auto CCpu = gCpu->cloneTensor(matmul->getOutput());
    // CCpu->printData();
    //  check results on CPU
    EXPECT_TRUE(CCpu->equalData(ansVec));
    // print a tensor/operator/graph by print()
    // gCuda->print();
}

TEST(Matmul, cuBlas) {
    testMatmulCuda(IncrementalGenerator(), OneGenerator(), false, false,
                   Shape{1, 3, 5}, Shape{1, 5, 2},
                   ExpectOutput{10, 10, 35, 35, 60, 60});
    testMatmulCuda(IncrementalGenerator(), IncrementalGenerator(), true, false,
                   Shape{2, 3, 4}, Shape{2, 3, 2},
                   ExpectOutput{40, 52, 46, 61, 52, 70, 58, 79, 400, 448, 424,
                                475, 448, 502, 472, 529});
}

TEST(Matmul, tune) {
    auto cpuRuntime = CpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(cpuRuntime);
    auto ACpu = gCpu->addTensor(Shape{1, 3, 5}, DataType::Float32);
    auto BCpu = gCpu->addTensor(Shape{1, 5, 2}, DataType::Float32);
    gCpu->dataMalloc();
    ACpu->setData(IncrementalGenerator());
    BCpu->setData(IncrementalGenerator());

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    auto gCuda = make_ref<GraphObj>(cudaRuntime);
    auto ACuda = gCuda->cloneTensor(ACpu);
    auto BCuda = gCuda->cloneTensor(BCpu);
    auto matmul = gCuda->addOp<MatmulObj>(ACuda, BCuda, nullptr);

    // allocate CUDA memory
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda, true);
}

}; // namespace infini