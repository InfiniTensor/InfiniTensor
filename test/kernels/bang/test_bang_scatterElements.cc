#include "bang/bang_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/scatterElements.h"
#include "test.h"
#include <cmath>

namespace infini {

void bangScatterElementsFp32(const Shape &inputShape,
                             const vector<float> &inputData,
                             const Shape &indicesShape,
                             const vector<int64_t> &indicesData,
                             const Shape &updatesShape,
                             const vector<float> &updatesData,
                             const vector<float> &outputData, int axis) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto bangRuntime = make_ref<BangRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu =
        make_ref<TensorObj>(inputShape, DataType::Float32, cpuRuntime);
    Tensor indicesCpu =
        make_ref<TensorObj>(indicesShape, DataType::Int64, cpuRuntime);
    Tensor updatesCpu =
        make_ref<TensorObj>(updatesShape, DataType::Float32, cpuRuntime);

    // GPU
    Graph bangGraph = make_ref<GraphObj>(bangRuntime);
    auto inputGpu = bangGraph->cloneTensor(inputCpu);
    auto indicesGpu = bangGraph->cloneTensor(indicesCpu);
    auto updatesGpu = bangGraph->cloneTensor(updatesCpu);

    auto gpuOp = bangGraph->addOp<ScatterElementsObj>(
        inputGpu, indicesGpu, updatesGpu, nullptr, axis);
    bangGraph->dataMalloc();
    inputGpu->copyin(inputData);
    indicesGpu->copyin(indicesData);
    updatesGpu->copyin(updatesData);

    bangRuntime->run(bangGraph);
    auto outputGpu = gpuOp->getOutput(0);
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    // outputGpu2Cpu->printData();
    //   Check
    EXPECT_TRUE(outputGpu2Cpu->equalData(outputData));
}
TEST(bangScatterElementsFp32, run) {
    bangScatterElementsFp32(
        Shape{3, 3}, vector<float>{0., 0., 0., 0., 0., 0., 0., 0., 0.},
        Shape{2, 3}, vector<int64_t>{1, 0, 2, 0, 2, 1}, Shape{2, 3},
        vector<float>{1.0000, 1.1000, 1.2000, 2.0000, 2.1000, 2.2000},
        vector<float>{2.0000, 1.1000, 0.0000, 1.0000, 0.0000, 2.2000, 0.0000,
                      2.1000, 1.2000},
        0);
    bangScatterElementsFp32(
        Shape{1, 5}, vector<float>{1., 2., 3., 4., 5.}, Shape{1, 2},
        vector<int64_t>{1, 3}, Shape{1, 2}, vector<float>{1.1000, 2.1000},
        vector<float>{1.0000, 1.1000, 3.0000, 2.1000, 5.0000}, 1);
}

} // namespace infini

