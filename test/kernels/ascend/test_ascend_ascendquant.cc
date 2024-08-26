#include "ascend/ascend_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/ascend_quant.h"

#include "test.h"

namespace infini {

template <class T> void testAscendquant(const Shape &shapeA) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto npuRuntime = make_ref<ASCENDRuntimeObj>();

    Tensor outputCpu = make_ref<TensorObj>(shapeA, DataType::Int8, cpuRuntime);
    outputCpu->dataMalloc();
    outputCpu->copyin(vector<int8_t>{1, 12, 28});

    // NPU
    Graph npuGraph = make_ref<GraphObj>(npuRuntime);
    auto inputNpu = npuGraph->addTensor(shapeA, DataType::Float32);

    auto npuOp =
        npuGraph->addOp<T>(inputNpu, nullptr, vector<float>{1.0, 2.0, 3.0},
                           vector<float>{0.3, 0.5, 0.7});
    npuGraph->dataMalloc();
    inputNpu->copyin(vector<float>{1, 6, 9});
    npuRuntime->run(npuGraph);

    auto outputNpu = npuOp->getOutput();
    auto outputNpu2Cpu = outputNpu->clone(cpuRuntime);

    // Check
    EXPECT_TRUE(outputCpu->equalData(outputNpu2Cpu, 1e-3));
}

TEST(ascend_Ascendquant, run) {
    aclInit(nullptr);
    testAscendquant<AscendQuantObj>(Shape{1, 3});
    aclFinalize();
}

} // namespace infini
