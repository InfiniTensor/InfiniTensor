#include "ascend/ascend_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/instance_norm.h"

#include "test.h"

namespace infini {

void test_instancenormFp32(const Shape &inputShape,
                           const vector<float> &inputData,
                           const Shape &scaleShape,
                           const vector<float> &scaleData, float eps,
                           const vector<float> &ExpectData,
                           const Shape &biasShape,
                           const vector<float> &biasData) {

    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto bias = gCpu->addTensor(biasShape, DataType::Float32);
    auto input = gCpu->addTensor(inputShape, DataType::Float32);
    auto scale = gCpu->addTensor(scaleShape, DataType::Float32);
    gCpu->dataMalloc();
    bias->copyin(biasData); //
    // bias->printData();
    input->copyin(inputData);
    scale->copyin(scaleData); //
    auto ascendRuntime = make_ref<ASCENDRuntimeObj>();
    Graph gAscend = make_ref<GraphObj>(ascendRuntime);
    auto biasNpu = gAscend->cloneTensor(bias);
    auto inputNpu = gAscend->cloneTensor(input);
    auto scaleNpu = gAscend->cloneTensor(scale);
    // gCpu->cloneTensor(biasNpu)->printData();
    auto op =
        gAscend->addOp<InstanceNormObj>(inputNpu, nullptr, scaleNpu, biasNpu,
                                        eps); // InstancenormObj
    gAscend->dataMalloc();
    biasNpu->copyin(biasData);
    // gCpu->cloneTensor(biasNpu)->printData();
    inputNpu->copyin(inputData);
    scaleNpu->copyin(scaleData);
    ascendRuntime->run(gAscend);

    auto oCpu = gCpu->cloneTensor(op->getOutput()); // move Data from npu to cpu
    oCpu->printData();                              //->printData
    EXPECT_TRUE(oCpu->equalData(ExpectData));
}

TEST(CUDA_InstancenormFp32, run) {
    aclInit(nullptr);
    test_instancenormFp32(
        Shape{2, 3, 2, 3},
        vector<float>{0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,
                      9.,  10., 11., 12., 13., 14., 15., 16., 17.,
                      18., 19., 20., 21., 22., 23., 24., 25., 26.,
                      27., 28., 29., 30., 31., 32., 33., 34., 35.},
        Shape{3}, vector<float>{0.3, 0.2, 0.5}, 1e-5,
        vector<float>{
            -0.3674207, 0.0000000, 0.6123678, -0.3674207, 0.0000000, 0.6123678,
            -0.3674207, 0.0000000, 0.6123678, -0.3674207, 0.0000000, 0.6123678,
            -0.3674207, 0.0000000, 0.6123678, -0.3674207, 0.0000000, 0.6123678,
            -0.3674207, 0.0000000, 0.6123678, -0.3674207, 0.0000000, 0.6123678,
            -0.3674207, 0.0000000, 0.6123678, -0.3674207, 0.0000000, 0.6123678,
            -0.3674207, 0.0000000, 0.6123678, -0.3674207, 0.0000000, 0.6123678},
        Shape{3}, vector<float>{0, 0, 0});

    aclFinalize();
} // python output

} // namespace infini
