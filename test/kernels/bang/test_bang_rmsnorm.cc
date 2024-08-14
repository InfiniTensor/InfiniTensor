#include "bang/bang_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/rms_norm.h"
#include "test.h"
#include <cmath>
double eps = 3e-3;
namespace infini {

void bangRMSNormFp32(const Shape &inputShape, const vector<float> &inputData,
                     const Shape &weightShape, const vector<float> &weightData,
                     const vector<float> &expectData) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto bangRuntime = make_ref<BangRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu =
        make_ref<TensorObj>(inputShape, DataType::Float32, cpuRuntime);
    Tensor weightCpu =
        make_ref<TensorObj>(weightShape, DataType::Float32, cpuRuntime);

    // GPU
    Graph bangGraph = make_ref<GraphObj>(bangRuntime);
    auto inputGpu = bangGraph->cloneTensor(inputCpu);
    auto weightGpu = bangGraph->cloneTensor(weightCpu);
    auto gpuOp = bangGraph->addOp<RMSNormObj>(inputGpu, weightGpu, nullptr);
    bangGraph->dataMalloc();
    inputGpu->copyin(inputData);
    weightGpu->copyin(weightData);
    bangRuntime->run(bangGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    // outputGpu2Cpu->printData();
    //   Check
    EXPECT_TRUE(outputGpu2Cpu->equalData(expectData, eps));
}
TEST(bangRMSNormFp32, run) {
    bangRMSNormFp32(Shape{2, 3, 2, 2},
                    vector<float>{0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,
                                  8.,  9.,  10., 11., 12., 13., 14., 15.,
                                  16., 17., 18., 19., 20., 21., 22., 23.},
                    Shape{2}, vector<float>{1., 2.},
                    vector<float>{0.000000, 2.828399, 0.784464, 2.353392,
                                  0.883452, 2.208630, 0.920358, 2.147502,
                                  0.939552, 2.113993, 0.951303, 2.092867,
                                  0.959233, 2.078338, 0.964944, 2.067737,
                                  0.969252, 2.059661, 0.972618, 2.053304,
                                  0.975320, 2.048172, 0.977536, 2.043940});
}

} // namespace infini
