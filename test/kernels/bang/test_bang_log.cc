#include "bang/bang_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

template <class T>
void testLog(const std::function<void(void *, size_t, DataType)> &generator,
             const Shape &shape, LogObj::LogType type) {
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
    auto gpuOp = bangGraph->addOp<T>(inputGpu, nullptr, type);
    bangGraph->dataMalloc();
    bangRuntime->run(bangGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    inputCpu->printData();
    outputGpu2Cpu->printData();
    EXPECT_TRUE(1);
}

TEST(cnnl_Log, run) {
    testLog<LogObj>(IncrementalGenerator(), Shape{1, 2, 2, 3}, LogObj::Log2);
    testLog<LogObj>(IncrementalGenerator(), Shape{1, 2, 2, 3}, LogObj::LogE);
    testLog<LogObj>(IncrementalGenerator(), Shape{1, 2, 2, 3}, LogObj::Log10);
}

} // namespace infini
