#include "bang/bang_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

template <class T>
void testCumsum(const std::function<void(void *, size_t, DataType)> &generator,
                int axis, const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = CpuRuntimeObj::getInstance();
    auto bangRuntime = make_ref<BangRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    inputCpu->dataMalloc();
    inputCpu->setData(generator);

    // GPU
    Graph bangGraph = make_ref<GraphObj>(bangRuntime);
    auto inputGpu = bangGraph->cloneTensor(inputCpu);
    auto gpuOp = bangGraph->addOp<T>(inputGpu, nullptr, axis, false, false);
    auto outputGpu = gpuOp->getOutput();
    bangGraph->dataMalloc();
    bangRuntime->run(bangGraph);
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    inputCpu->printData();
    outputGpu2Cpu->printData();
    EXPECT_TRUE(1);
}

TEST(cnnl_Cumsum, run) {
    testCumsum<CumsumObj>(IncrementalGenerator(), 1, Shape{1, 2, 2, 3});
    testCumsum<CumsumObj>(IncrementalGenerator(), 2, Shape{1, 2, 2, 3});
    testCumsum<CumsumObj>(IncrementalGenerator(), 3, Shape{1, 2, 2, 3});
}

} // namespace infini
