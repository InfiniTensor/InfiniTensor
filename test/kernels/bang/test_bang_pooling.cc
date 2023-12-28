#include "bang/bang_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/pooling.h"

#include "test.h"

namespace infini {

template <class T, typename std::enable_if<std::is_base_of<PoolingObj, T>{},
                                           int>::type = 0>
void testPooling(const std::function<void(void *, size_t, DataType)> &generator,
                 const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto bangRuntime = make_ref<BangRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    Graph cpuGraph = make_ref<GraphObj>(cpuRuntime);
    auto cpuOp =
        cpuGraph->addOp<T>(inputCpu, nullptr, 3, 3, 1, 1, 1, 1, 2, 2, 0);
    cpuGraph->addTensor(inputCpu);
    cpuGraph->dataMalloc();
    inputCpu->setData(generator);
    cpuRuntime->run(cpuGraph);
    auto outputCpu = cpuOp->getOutput();

    // GPU
    Graph bangGraph = make_ref<GraphObj>(bangRuntime);
    auto inputGpu = bangGraph->cloneTensor(inputCpu);
    auto gpuOp =
        bangGraph->addOp<T>(inputGpu, nullptr, 3, 3, 1, 1, 1, 1, 2, 2, 0);
    bangGraph->dataMalloc();
    inputGpu->setData(generator);
    bangRuntime->run(bangGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    EXPECT_TRUE(1);
}

TEST(cnnl_Pooling, run) {
    testPooling<MaxPoolObj>(IncrementalGenerator(), Shape{1, 3, 5, 5});
    testPooling<AvgPoolObj>(IncrementalGenerator(), Shape{1, 3, 5, 5});
}

} // namespace infini
