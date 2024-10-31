#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/gemm.h"

#include "test.h"

namespace infini {
using ExpectOutput = vector<float>;

void testGemmCuda(
    const std::function<void(void *, size_t, DataType)> &generatorA,
    const std::function<void(void *, size_t, DataType)> &generatorB,
    const std::function<void(void *, size_t, DataType)> &generatorC,
    float alpha, float beta, bool transA, bool transB, const Shape &shapeA,
    const Shape &shapeB, const Shape &shapeC, const ExpectOutput &ansVec) {

    auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(cpuRuntime);
    auto ACpu = gCpu->addTensor(shapeA, DataType::Float32);
    auto BCpu = gCpu->addTensor(shapeB, DataType::Float32);
    auto CCpu = gCpu->addTensor(shapeC, DataType::Float32);
    gCpu->dataMalloc();
    ACpu->setData(generatorA);
    BCpu->setData(generatorB);
    CCpu->setData(generatorC);

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    auto gCuda = make_ref<GraphObj>(cudaRuntime);
    auto ACuda = gCuda->cloneTensor(ACpu);
    auto BCuda = gCuda->cloneTensor(BCpu);
    auto CCuda = gCuda->cloneTensor(CCpu);
    auto gemm = gCuda->addOp<GemmObj>(ACuda, BCuda, nullptr, CCuda, alpha, beta,
                                      transA, transB);

    // allocate CUDA memory
    gCuda->dataMalloc();
    ACuda->setData(generatorA);
    BCuda->setData(generatorB);
    CCuda->setData(generatorC);
    cudaRuntime->run(gCuda);

    auto YCpu = gCpu->cloneTensor(gemm->getOutput());
    EXPECT_TRUE(YCpu->equalData(ansVec));
}

TEST(CUDA_Gemm, run) {
    testGemmCuda(IncrementalGenerator(), OneGenerator(), OneGenerator(), 1.0,
                 0.0, false, false, Shape{3, 5}, Shape{5, 2}, Shape{1},
                 ExpectOutput{10, 10, 35, 35, 60, 60});
}

}; // namespace infini
