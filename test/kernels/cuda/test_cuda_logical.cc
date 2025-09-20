#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/element_wise.h"
#include "operators/logical.h"

#include "test.h"
#include "utils/data_generator.h"
#include <cstdint>

// Unit tests for CUDA logical/bitwise kernels. Tests use simple deterministic
// generators (Incremental/Val/Random) and compare GPU results against CPU.

namespace infini {

template <class T>
void testLogicalBinaryBoolCuda(
    const std::function<void(void *, size_t, DataType)> &generatorA,
    const std::function<void(void *, size_t, DataType)> &generatorB,
    const Shape &shape, const vector<uint8_t> &ansVec) {
    auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto gCpu = make_ref<GraphObj>(cpuRuntime);
    auto aCpu = gCpu->addTensor(shape, DataType::Bool);
    auto bCpu = gCpu->addTensor(shape, DataType::Bool);
    gCpu->dataMalloc();
    aCpu->setData(generatorA);
    bCpu->setData(generatorB);
    // aCpu->printData();
    // bCpu->printData();

    // cuda graph
    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    auto gCuda = make_ref<GraphObj>(cudaRuntime);
    auto aGpu = gCuda->cloneTensor(aCpu);
    auto bGpu = gCuda->cloneTensor(bCpu);
    auto op = gCuda->addOp<T>(aGpu, bGpu, nullptr);
    gCuda->dataMalloc();
    aGpu->setData(generatorA);
    bGpu->setData(generatorB);

    cudaRuntime->run(gCuda);

    auto cCpu = gCpu->cloneTensor(op->getOutput());
    // cCpu->printData();
    EXPECT_TRUE(cCpu->equalData(ansVec));
}

template <class T>
void testLogicalUnaryBoolCuda(
    const std::function<void(void *, size_t, DataType)> &generatorA,
    const Shape &shape, const vector<uint8_t> &ansVec) {
    auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto gCpu = make_ref<GraphObj>(cpuRuntime);
    auto aCpu = gCpu->addTensor(shape, DataType::Bool);
    gCpu->dataMalloc();
    aCpu->setData(generatorA);
    // aCpu->printData();

    // cuda graph
    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    auto gCuda = make_ref<GraphObj>(cudaRuntime);
    auto aGpu = gCuda->cloneTensor(aCpu);
    auto op = gCuda->addOp<T>(aGpu, nullptr);
    gCuda->dataMalloc();
    aGpu->setData(generatorA);

    cudaRuntime->run(gCuda);

    auto cCpu = gCpu->cloneTensor(op->getOutput());
    // cCpu->printData();
    EXPECT_TRUE(cCpu->equalData(ansVec));
}

template <class T>
void testLogicalBinaryIntCuda(
    const std::function<void(void *, size_t, DataType)> &generatorA,
    const std::function<void(void *, size_t, DataType)> &generatorB,
    const Shape &shape, const vector<uint32_t> &ansVec) {
    auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto gCpu = make_ref<GraphObj>(cpuRuntime);
    auto aCpu = gCpu->addTensor(shape, DataType::UInt32);
    auto bCpu = gCpu->addTensor(shape, DataType::UInt32);
    gCpu->dataMalloc();
    aCpu->setData(generatorA);
    bCpu->setData(generatorB);
    // aCpu->printData();
    // bCpu->printData();

    // cuda graph
    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    auto gCuda = make_ref<GraphObj>(cudaRuntime);
    auto aGpu = gCuda->cloneTensor(aCpu);
    auto bGpu = gCuda->cloneTensor(bCpu);
    auto op = gCuda->addOp<T>(aGpu, bGpu, nullptr);
    gCuda->dataMalloc();
    aGpu->setData(generatorA);
    bGpu->setData(generatorB);

    cudaRuntime->run(gCuda);

    auto cCpu = gCpu->cloneTensor(op->getOutput());
    // cCpu->printData();
    EXPECT_TRUE(cCpu->equalData(ansVec));
}

template <class T>
void testLogicalUnaryIntCuda(
    const std::function<void(void *, size_t, DataType)> &generatorA,
    const Shape &shape, const vector<uint32_t> &ansVec) {
    auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto gCpu = make_ref<GraphObj>(cpuRuntime);
    auto aCpu = gCpu->addTensor(shape, DataType::UInt32);
    gCpu->dataMalloc();
    aCpu->setData(generatorA);
    // aCpu->printData();

    // cuda graph
    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    auto gCuda = make_ref<GraphObj>(cudaRuntime);
    auto aGpu = gCuda->cloneTensor(aCpu);
    auto op = gCuda->addOp<T>(aGpu, nullptr);
    gCuda->dataMalloc();
    aGpu->setData(generatorA);

    cudaRuntime->run(gCuda);

    auto cCpu = gCpu->cloneTensor(op->getOutput());
    // cCpu->printData();
    EXPECT_TRUE(cCpu->equalData(ansVec));
}

TEST(cuda_logical_unary_bool, run) {
    testLogicalUnaryBoolCuda<NotObj>(ZeroGenerator(), Shape{1, 2, 2, 3},
                                     std::vector<uint8_t>(12, 1));
}

TEST(cuda_logical_binary_bool, run) {
    testLogicalBinaryBoolCuda<AndObj>(OneGenerator(), OneGenerator(),
                                      Shape{1, 2, 2, 3},
                                      std::vector<uint8_t>(12, 1));
    testLogicalBinaryBoolCuda<OrObj>(ZeroGenerator(), OneGenerator(),
                                     Shape{1, 2, 2, 3},
                                     std::vector<uint8_t>(12, 1));
    testLogicalBinaryBoolCuda<XorObj>(ZeroGenerator(), OneGenerator(),
                                      Shape{1, 2, 2, 3},
                                      std::vector<uint8_t>(12, 1));
}

TEST(cuda_logical_binary_int, run) {
    testLogicalBinaryIntCuda<BitAndObj>(
        IncrementalGenerator(), IncrementalGenerator(), Shape{1, 2, 2, 3},
        std::vector<uint32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    testLogicalBinaryIntCuda<BitOrObj>(
        IncrementalGenerator(), IncrementalGenerator(), Shape{1, 2, 2, 3},
        std::vector<uint32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    testLogicalBinaryIntCuda<BitXorObj>(
        IncrementalGenerator(), IncrementalGenerator(), Shape{1, 2, 2, 3},
        std::vector<uint32_t>(12, 0));
    testLogicalBinaryIntCuda<BitLeftShiftObj>(
        IncrementalGenerator(), OneGenerator(), Shape{1, 2, 2, 3},
        std::vector<uint32_t>{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22});
    testLogicalBinaryIntCuda<BitRightShiftObj>(
        IncrementalGenerator(), OneGenerator(), Shape{1, 2, 2, 3},
        std::vector<uint32_t>{0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5});
}

TEST(cuda_logical_unary_int, run) {
    testLogicalUnaryIntCuda<BitNotObj>(
        IncrementalGenerator(), Shape{1, 2, 2, 3},
        std::vector<uint32_t>{4294967295, 4294967294, 4294967293, 4294967292,
                              4294967291, 4294967290, 4294967289, 4294967288,
                              4294967287, 4294967286, 4294967285, 4294967284});
}

} // namespace infini
