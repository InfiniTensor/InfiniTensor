#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "kunlun/kunlun_runtime.h"
#include "operators/matmul.h"

#include "test.h"

namespace infini {
using ExpectOutput = vector<float>;

void testMatmulKUNLUNWithBias(
    const std::function<void(void *, size_t, DataType)> &generatorA,
    const std::function<void(void *, size_t, DataType)> &generatorB,
    const std::function<void(void *, size_t, DataType)> &generatorBias,
    bool transA, bool transB, const Shape &shapeA, const Shape &shapeB,
    const Shape &shapeBias, const ExpectOutput &ansVec) {
    auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(cpuRuntime);
    auto ACpu = gCpu->addTensor(shapeA, DataType::Float32);
    auto BCpu = gCpu->addTensor(shapeB, DataType::Float32);
    auto BiasCpu = gCpu->addTensor(shapeBias, DataType::Float32);
    gCpu->dataMalloc();
    ACpu->setData(generatorA);
    BCpu->setData(generatorB);
    BiasCpu->setData(generatorBias);

    auto kunlunRuntime = make_ref<KUNLUNRuntimeObj>();
    auto gKunlun = make_ref<GraphObj>(kunlunRuntime);
    auto AKunlun = gKunlun->cloneTensor(ACpu);
    auto BKunlun = gKunlun->cloneTensor(BCpu);
    auto BiasKunlun = gKunlun->cloneTensor(BiasCpu);
    auto matmul = gKunlun->addOp<MatmulObj>(AKunlun, BKunlun, nullptr, transA,
                                            transB, BiasKunlun);

    // allocate Kunlun memory
    gKunlun->dataMalloc();
    AKunlun->setData(generatorA);
    BKunlun->setData(generatorB);
    BiasKunlun->setData(generatorBias);
    kunlunRuntime->run(gKunlun);

    auto CCpu = gCpu->cloneTensor(matmul->getOutput());
    // CCpu->printData();
    //  check results on CPU
    EXPECT_TRUE(CCpu->equalData(ansVec));
    // print a tensor/operator/graph by print()
    // gKunlun->print();
}

void testMatmulKUNLUN(
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

    auto kunlunRuntime = make_ref<KUNLUNRuntimeObj>();
    auto gKunlun = make_ref<GraphObj>(kunlunRuntime);
    auto AKunlun = gKunlun->cloneTensor(ACpu);
    auto BKunlun = gKunlun->cloneTensor(BCpu);
    auto matmul = gKunlun->addOp<MatmulObj>(AKunlun, BKunlun, nullptr, transA,
                                            transB, nullptr);

    // allocate Kunlun memory
    gKunlun->dataMalloc();
    AKunlun->setData(generatorA);
    BKunlun->setData(generatorB);
    kunlunRuntime->run(gKunlun);

    auto CCpu = gCpu->cloneTensor(matmul->getOutput());
    // CCpu->printData();
    //  check results on CPU
    EXPECT_TRUE(CCpu->equalData(ansVec));
    // print a tensor/operator/graph by print()
    // gKunlun->print();
}

TEST(XDNN_Matmul, run) {
    testMatmulKUNLUN(IncrementalGenerator(), OneGenerator(), false, false,
                     Shape{1, 3, 5}, Shape{1, 5, 2},
                     ExpectOutput{10, 10, 35, 35, 60, 60});
    testMatmulKUNLUN(IncrementalGenerator(), IncrementalGenerator(), true,
                     false, Shape{2, 3, 4}, Shape{2, 3, 2},
                     ExpectOutput{40, 52, 46, 61, 52, 70, 58, 79, 400, 448, 424,
                                  475, 448, 502, 472, 529});
    testMatmulKUNLUN(IncrementalGenerator(), IncrementalGenerator(), false,
                     false, Shape{3, 5}, Shape{5, 2},
                     ExpectOutput{60, 70, 160, 195, 260, 320});
}

TEST(XDNN_Matmul_With_Bias, run) {
    testMatmulKUNLUNWithBias(IncrementalGenerator(), OneGenerator(),
                             OneGenerator(), false, false, Shape{1, 3, 5},
                             Shape{1, 5, 2}, Shape{2},
                             ExpectOutput{11, 11, 36, 36, 61, 61});
    testMatmulKUNLUNWithBias(IncrementalGenerator(), IncrementalGenerator(),
                             OneGenerator(), true, false, Shape{2, 3, 4},
                             Shape{2, 3, 2}, Shape{4, 2},
                             ExpectOutput{41, 53, 47, 62, 53, 71, 59, 80, 401,
                                          449, 425, 476, 449, 503, 473, 530});
}

}; // namespace infini
