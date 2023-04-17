
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "intelcpu/mkl_runtime.h"
#include "operators/matmul.h"

#include "test.h"

namespace infini {
using ExpectOutput = vector<float>;

void testMatmulMkl(
    const std::function<void(void *, size_t, DataType)> &generatorA,
    const std::function<void(void *, size_t, DataType)> &generatorB,
    bool transA, bool transB, const Shape &shapeA, const Shape &shapeB,
    const ExpectOutput &ansVec) {
    auto cpuRuntime = MklRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(cpuRuntime);
    auto ACpu = gCpu->addTensor(shapeA, DataType::Float32);
    auto BCpu = gCpu->addTensor(shapeB, DataType::Float32);
    gCpu->dataMalloc();
    ACpu->setData(generatorA);
    BCpu->setData(generatorB);

    auto matmul = gCpu->addOp<MatmulObj>(ACpu, BCpu, nullptr, transA, transB);

    gCpu->dataMalloc();
    cpuRuntime->run(gCpu);
    EXPECT_TRUE(matmul->getOutput()->equalData(ansVec));
}

TEST(mkl_Matmul, run) {
    testMatmulMkl(IncrementalGenerator(), OneGenerator(), false, false,
                  Shape{1, 3, 5}, Shape{1, 5, 2},
                  ExpectOutput{10, 10, 35, 35, 60, 60});
    testMatmulMkl(IncrementalGenerator(), IncrementalGenerator(), true, false,
                  Shape{2, 3, 4}, Shape{2, 3, 2},
                  ExpectOutput{40, 52, 46, 61, 52, 70, 58, 79, 400, 448, 424,
                               475, 448, 502, 472, 529});
}

}; // namespace infini
