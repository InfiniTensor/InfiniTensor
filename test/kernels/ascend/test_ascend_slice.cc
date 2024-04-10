#include "ascend/ascend_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/slice.h"

#include "test.h"

namespace infini {

TEST(ascend_Unary, run) {
    aclInit(nullptr);
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto npuRuntime = make_ref<ASCENDRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu =
        make_ref<TensorObj>(Shape{3, 2, 1, 5}, DataType::Float32, cpuRuntime);
    // inputCpu->dataMalloc();
    // inputCpu->setData(IncrementalGenerator());

    // NPU
    Graph npuGraph = make_ref<GraphObj>(npuRuntime);
    auto inputNpu = npuGraph->cloneTensor(inputCpu);
    auto npuOp = npuGraph->addOp<SliceObj>(inputNpu, nullptr, vector<int>{1, 1},
                                           vector<int>{2, 5}, vector<int>{0, 3},
                                           std::nullopt);
    npuGraph->dataMalloc();
    inputNpu->setData(IncrementalGenerator());
    npuRuntime->run(npuGraph);
    auto outputNpu = npuOp->getOutput();
    auto outputNpu2Cpu = outputNpu->clone(cpuRuntime);

    // Check
    EXPECT_TRUE(outputNpu2Cpu->equalData(
        vector<float>{11, 12, 13, 14, 16, 17, 18, 19}));
    aclFinalize();
}

} // namespace infini
