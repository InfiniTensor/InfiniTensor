#include "ascend/ascend_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/ascend_dequant.h"

#include "test.h"

namespace infini {

template <class T> void testAscenddequant(const Shape &shapeA) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    // Graph gCpu = make_ref<GraphObj>(cpuRuntime);
    auto npuRuntime = make_ref<ASCENDRuntimeObj>();

    // Tensor outputCpu =
    //     make_ref<TensorObj>(shapeA, DataType::Float32, cpuRuntime);
    // outputCpu->dataMalloc();
    // outputCpu->copyin(vector<float>{1, 6, 9});

    // NPU
    Graph npuGraph = make_ref<GraphObj>(npuRuntime);
    auto inputNpu = npuGraph->addTensor(shapeA, DataType::Int8);

    auto npuOp =
        npuGraph->addOp<T>(inputNpu, nullptr, vector<float>{1.0, 2.0, 3.0},
                           vector<float>{1.0, 2.0, 3.0});
    npuGraph->dataMalloc();
    inputNpu->copyin(vector<int8_t>{1, 6, 9});
    npuRuntime->run(npuGraph);

    auto outputNpu = npuOp->getOutput();
    auto outputNpu2Cpu = outputNpu->clone(cpuRuntime);
    // auto o0Cpu = gCpu->cloneTensor(outputNpu);

    // Check
    EXPECT_TRUE(outputNpu2Cpu->equalData(vector<float>{2.0, 16.0, 36.0}));
}

TEST(ascend_Ascenddequant, run) {
    aclInit(nullptr);
    testAscenddequant<AscendDequantObj>(Shape{1, 3});
    aclFinalize();
}

} // namespace infini
