#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/matmul.h"

#include "test.h"

namespace infini {
using ExpectOutput = vector<float>;

void testMatmulCuda(
    const std::function<void(void *, size_t, DataType)> &generatorA,
    const std::function<void(void *, size_t, DataType)> &generatorB,
    bool transA, bool transB, const Shape &shapeA, const Shape &shapeB,
    const ExpectOutput &ansVec) {
    auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
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

TEST(cuBLAS_Matmul, run) {
    testMatmulCuda(IncrementalGenerator(), OneGenerator(), false, false,
                   Shape{1, 3, 5}, Shape{1, 5, 2},
                   ExpectOutput{10, 10, 35, 35, 60, 60});
    testMatmulCuda(IncrementalGenerator(), IncrementalGenerator(), true, false,
                   Shape{2, 3, 4}, Shape{2, 3, 2},
                   ExpectOutput{40, 52, 46, 61, 52, 70, 58, 79, 400, 448, 424,
                                475, 448, 502, 472, 529});
    testMatmulCuda(
        IncrementalGenerator(), IncrementalGenerator(), false, false,
        Shape{2, 3, 5}, Shape{5, 2},
        ExpectOutput{60, 70, 160, 195, 260, 320, 360, 445, 460, 570, 560, 695});
    testMatmulCuda(IncrementalGenerator(), IncrementalGenerator(), true, false,
                   Shape{2, 5, 3}, Shape{5, 2},
                   ExpectOutput{180, 210, 200, 235, 220, 260, 480, 585, 500,
                                610, 520, 635});
    testMatmulCuda(IncrementalGenerator(), IncrementalGenerator(), false, false,
                   Shape{3, 5}, Shape{5, 2},
                   ExpectOutput{60, 70, 160, 195, 260, 320});
}

TEST(cuBLAS_Matmul, tune) {
    // Matmul([A^T,B,act=0],A=597,B=595,C=598,bmnk=[1,4,4096,448])
    const int B = 1, M = 4, N = 4096, K = 448;
    const bool transA = true, transB = false;
    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph g = make_ref<GraphObj>(cudaRuntime);
    auto a = g->addTensor(transA ? Shape{B, K, M} : Shape{B, M, K});
    auto b = g->addTensor(transB ? Shape{B, N, K} : Shape{B, K, N});
    // allocate CUDA memory
    g->dataMalloc();
    a->setData(IncrementalGenerator());
    b->setData(IncrementalGenerator());

    auto matmul = g->addOp<MatmulObj>(a, b, nullptr, transA, transB);
    matmul->print();
    double time = cudaRuntime->getPerfTime(g);
    EXPECT_GT(time, 1e-3);
    EXPECT_LT(time, 1);
    cudaRuntime->run(g, true);
}

}; // namespace infini
