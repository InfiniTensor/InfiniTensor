#include "ascend/ascend_runtime.h"
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
    auto npuRuntime = make_ref<ASCENDRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    inputCpu->dataMalloc();
    inputCpu->setData(generator);

    // GPU
    Graph npuGraph = make_ref<GraphObj>(npuRuntime);
    auto inputNpu = npuGraph->cloneTensor(inputCpu);
    auto npuOp =
        npuGraph->addOp<T>(inputNpu, nullptr, 3, 3, 1, 1, 1, 1, 2, 2, 0);
    npuGraph->dataMalloc();
    inputNpu->setData(generator);
    npuRuntime->run(npuGraph);

    auto outputNpu = npuOp->getOutput();
    auto outputNpu2Cpu = outputNpu->clone(cpuRuntime);
    inputCpu->printData();
    outputNpu2Cpu->printData();
    EXPECT_TRUE(1);
}

TEST(cnnl_Pooling, run) {
    aclInit(nullptr);
    testPooling<MaxPoolObj>(IncrementalGenerator(), Shape{1, 2, 5, 5});
    testPooling<AvgPoolObj>(IncrementalGenerator(), Shape{1, 2, 5, 5});
    aclFinalize();
}

} // namespace infini
