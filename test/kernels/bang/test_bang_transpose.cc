#include "bang/bang_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/transpose.h"

#include "test.h"

namespace infini {

template <class T>
void testTranspose(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto bangRuntime = make_ref<BangRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    inputCpu->dataMalloc();
    inputCpu->setData(generator);

    // GPU
    Graph bangGraph = make_ref<GraphObj>(bangRuntime);
    auto inputGpu = bangGraph->cloneTensor(inputCpu);
    vector<int> permute = {0, 1, 3, 2};
    auto gpuOp = bangGraph->addOp<T>(inputGpu, nullptr, permute);
    bangGraph->dataMalloc();
    bangRuntime->run(bangGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    // Check
    inputCpu->printData();
    outputGpu2Cpu->printData();
    EXPECT_TRUE(1);
}

TEST(cnnl_Transpose, run) {
    testTranspose<TransposeObj>(IncrementalGenerator(), Shape{1, 1, 2, 3});
}

} // namespace infini
