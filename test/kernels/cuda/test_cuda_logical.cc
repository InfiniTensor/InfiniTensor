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
void testLogicalBinaryCuda(
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
void testLogicalUnaryCuda(
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

TEST(cuda_logical_unary, run) {
    testLogicalUnaryCuda<NotObj>(ZeroGenerator(), Shape{1, 2, 2, 3},
                                     std::vector<uint8_t>(12, 1));
}

TEST(cuda_logical_binary, run) {
    testLogicalBinaryCuda<AndObj>(OneGenerator(), OneGenerator(),
                                      Shape{1, 2, 2, 3},
                                      std::vector<uint8_t>(12, 1));
    testLogicalBinaryCuda<OrObj>(ZeroGenerator(), OneGenerator(),
                                     Shape{1, 2, 2, 3},
                                     std::vector<uint8_t>(12, 1));
    testLogicalBinaryCuda<XorObj>(ZeroGenerator(), OneGenerator(),
                                      Shape{1, 2, 2, 3},
                                      std::vector<uint8_t>(12, 1));
}

} // namespace infini
