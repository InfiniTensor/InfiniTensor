#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/dynamic_quantize_linear.h"

#include "test.h"

namespace infini {

void test_dynamicquantizeLinearFp32(
    const Shape &inputShape, const vector<float> &inputData,
    const vector<uint8_t> &outputYData, const vector<float> &outputYScaleData,
    const vector<uint8_t> &outputYZeroPointData) {

    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor(inputShape, DataType::Float32);

    gCpu->dataMalloc();

    input->copyin(inputData);

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto inputGpu = gCuda->cloneTensor(input);

    auto op = gCuda->addOp<DynamicQuantizeLinearObj>(
        inputGpu,
        std::nullopt); // DynamicQuantizeLinear
    gCuda->dataMalloc();

    inputGpu->copyin(inputData);

    cudaRuntime->run(gCuda);

    EXPECT_EQ(op->getOutputs().size(), (size_t)3);
    auto o0Cpu = gCpu->cloneTensor(op->getOutput(0));
    auto o1Cpu = gCpu->cloneTensor(op->getOutput(1));
    auto o2Cpu = gCpu->cloneTensor(op->getOutput(2));

    EXPECT_TRUE(o0Cpu->equalData(outputYData));
    EXPECT_TRUE(o1Cpu->equalData(outputYScaleData));
    EXPECT_TRUE(o2Cpu->equalData(outputYZeroPointData));
}

TEST(CUDA_DynamicquantizeLinearFp32, run) {

    test_dynamicquantizeLinearFp32(
        Shape{2, 3, 2, 3},
        vector<float>{0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,
                      9.,  10., 11., 12., 13., 14., 15., 16., 17.,
                      18., 19., 20., 21., 22., 23., 24., 25., 26.,
                      27., 28., 29., 30., 31., 32., 33., 34., 35.},
        vector<uint8_t>{0,   7,   15,  22,  29,  36,  44,  51,  58,
                        66,  73,  80,  87,  95,  102, 109, 117, 124,
                        131, 138, 146, 153, 160, 168, 175, 182, 189,
                        197, 204, 211, 219, 226, 233, 240, 248, 255},
        vector<float>{0.1372549}, vector<uint8_t>{0});
    test_dynamicquantizeLinearFp32(
        Shape{2, 3, 2, 2},
        vector<float>{0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,
                      8.,  9.,  10., 11., 12., 13., 14., 15.,
                      16., 17., 18., 19., 20., 21., 22., 23.},
        vector<uint8_t>{0,   11,  22,  33,  44,  55,  67,  78,
                        89,  100, 111, 122, 133, 144, 155, 166,
                        177, 188, 200, 211, 222, 233, 244, 255},
        vector<float>{0.0901961}, vector<uint8_t>{0});

} // python output

} // namespace infini
