#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "kunlun/kunlun_runtime.h"
#include "operators/transpose.h"

#include "test.h"

namespace infini {

template <class T>
void testTranspose(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto xpuRuntime = make_ref<KUNLUNRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    inputCpu->dataMalloc();
    inputCpu->setData(generator);

    // GPU
    Graph xpuGraph = make_ref<GraphObj>(xpuRuntime);
    auto inputGpu = xpuGraph->cloneTensor(inputCpu);
    vector<int> permute = {0, 1, 3, 2};
    auto gpuOp = xpuGraph->addOp<T>(inputGpu, nullptr, permute);
    xpuGraph->dataMalloc();
    xpuRuntime->run(xpuGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    // Check
    inputCpu->printData();
    outputGpu2Cpu->printData();
    EXPECT_TRUE(1);
}

TEST(xpu_Transpose, run) {
    testTranspose<TransposeObj>(IncrementalGenerator(), Shape{1, 1, 2, 3});
}

} // namespace infini
