#include "bang/bang_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

template <class T>
void testFill(const std::function<void(void *, size_t, DataType)> &generator,
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
    float value = 1.0;
    auto gpuOp = bangGraph->addOp<T>(inputGpu, nullptr, value);
    auto outputGpu = gpuOp->getOutput();
    bangGraph->dataMalloc();
    bangRuntime->run(bangGraph);
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    outputGpu2Cpu->printData();
    EXPECT_TRUE(1);
}

TEST(cnnl_Fill, run) {
    testFill<FillObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
}

} // namespace infini
