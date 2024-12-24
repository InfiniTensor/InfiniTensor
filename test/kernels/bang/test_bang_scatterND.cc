#include "bang/bang_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/scatterND.h"
#include "test.h"
#include <cmath>

namespace infini {

void bangScatterNDFp32(const Shape &inputShape, const vector<float> &inputData,
                       const Shape &indicesShape,
                       const vector<int64_t> &indicesData,
                       const Shape &updatesShape,
                       const vector<float> &updatesData,
                       const vector<float> &outputData) {
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

    auto gpuOp = bangGraph->addOp<ScatterNDObj>(inputGpu, indicesGpu,
                                                updatesGpu, nullptr);
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
TEST(bangScatterNDFp32, run) {
    bangScatterNDFp32(
        Shape{4, 4, 4},
        vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
                      1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
                      8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8,
                      8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8},
        Shape{2, 1}, vector<int64_t>{0, 2}, Shape{2, 4, 4},
        vector<float>{5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
                      1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4},
        vector<float>{5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
                      1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
                      1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
                      8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8});
    bangScatterNDFp32(Shape{8}, vector<float>{1, 2, 3, 4, 5, 6, 7, 8},
                      Shape{4, 1}, vector<int64_t>{4, 3, 1, 7}, Shape{4},
                      vector<float>{9, 10, 11, 12},
                      vector<float>{1, 11, 3, 10, 9, 6, 7, 12});
}

} // namespace infini

