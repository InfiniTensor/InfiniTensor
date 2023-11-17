#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/layer_norm.h"

#include "test.h"

namespace infini {

void test_layernorm(
    const Shape &inputShape, const vector<float> &inputData,
    const Shape &scaleShape, const vector<float> &scaleData, float eps,
    int axis, int stash_type, const vector<float> &ExpectData,
    const std::optional<Shape> &bShape = std::nullopt,
    const std::optional<std::vector<float>> &biasData = std::nullopt) {

    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    if (bShape.has_value() && biasData.has_value()) {
        Shape biasShape = *bShape;

        auto bias = gCpu->addTensor(biasShape, DataType::Float32);
        auto input = gCpu->addTensor(inputShape, DataType::Float32);
        auto scale = gCpu->addTensor(scaleShape, DataType::Float32);
        gCpu->dataMalloc();
        bias->copyin(*biasData); //
        // bias->printData();
        input->copyin(inputData);
        scale->copyin(scaleData); //
        auto cudaRuntime = make_ref<CudaRuntimeObj>();
        Graph gCuda = make_ref<GraphObj>(cudaRuntime);
        auto biasGpu = gCuda->cloneTensor(bias);
        auto inputGpu = gCuda->cloneTensor(input);
        auto scaleGpu = gCuda->cloneTensor(scale);
        // gCpu->cloneTensor(biasGpu)->printData();
        auto op =
            gCuda->addOp<LayerNormObj>(inputGpu, scaleGpu, nullptr, biasGpu,
                                       eps, axis, stash_type); // LayernormObj
        gCuda->dataMalloc();
        biasGpu->copyin(*biasData);
        // gCpu->cloneTensor(biasGpu)->printData();
        inputGpu->copyin(inputData);
        scaleGpu->copyin(scaleData);
        cudaRuntime->run(gCuda);

        auto oCpu =
            gCpu->cloneTensor(op->getOutput()); // move Data from gpu to cpu
        oCpu->printData();                      //->printData
        EXPECT_TRUE(oCpu->equalData(ExpectData));
    } else {

        auto input = gCpu->addTensor(inputShape, DataType::Float32);
        auto scale = gCpu->addTensor(scaleShape, DataType::Float32);
        gCpu->dataMalloc();

        input->copyin(inputData);
        scale->copyin(scaleData); //
        auto cudaRuntime = make_ref<CudaRuntimeObj>();
        Graph gCuda = make_ref<GraphObj>(cudaRuntime);

        auto inputGpu = gCuda->cloneTensor(input);
        auto scaleGpu = gCuda->cloneTensor(scale);
        auto op =
            gCuda->addOp<LayerNormObj>(inputGpu, scaleGpu, nullptr, nullptr,
                                       eps, axis, stash_type); // LayernormObj
        gCuda->dataMalloc();

        inputGpu->copyin(inputData);
        scaleGpu->copyin(scaleData);
        cudaRuntime->run(gCuda);

        auto oCpu =
            gCpu->cloneTensor(op->getOutput()); // move Data from gpu to cpu
        oCpu->printData();                      //->printData
        EXPECT_TRUE(oCpu->equalData(ExpectData));
    }
}

TEST(CUDA_Layernorm, run) {
    test_layernorm(
        Shape{2, 3, 2, 3},
        vector<float>{0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,
                      9.,  10., 11., 12., 13., 14., 15., 16., 17.,
                      18., 19., 20., 21., 22., 23., 24., 25., 26.,
                      27., 28., 29., 30., 31., 32., 33., 34., 35.},
        Shape{3}, vector<float>{0.3, 0.2, 0.5}, 1e-5, 3, 1,
        vector<float>{
            -0.3674207, 0.0000000, 0.6123678, -0.3674207, 0.0000000, 0.6123678,
            -0.3674207, 0.0000000, 0.6123678, -0.3674207, 0.0000000, 0.6123678,
            -0.3674207, 0.0000000, 0.6123678, -0.3674207, 0.0000000, 0.6123678,
            -0.3674207, 0.0000000, 0.6123678, -0.3674207, 0.0000000, 0.6123678,
            -0.3674207, 0.0000000, 0.6123678, -0.3674207, 0.0000000, 0.6123678,
            -0.3674207, 0.0000000, 0.6123678, -0.3674207, 0.0000000, 0.6123678},
        Shape{3}, vector<float>{0, 0, 0});
    test_layernorm(
        Shape{2, 3, 2, 3},
        vector<float>{0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,
                      9.,  10., 11., 12., 13., 14., 15., 16., 17.,
                      18., 19., 20., 21., 22., 23., 24., 25., 26.,
                      27., 28., 29., 30., 31., 32., 33., 34., 35.},
        Shape{3}, vector<float>{0.3, 0.2, 0.5}, 1e-5, 3, 1,
        vector<float>{
            -0.0674207, 0.2000000, 1.1123679, -0.0674207, 0.2000000, 1.1123679,
            -0.0674207, 0.2000000, 1.1123679, -0.0674207, 0.2000000, 1.1123679,
            -0.0674207, 0.2000000, 1.1123679, -0.0674207, 0.2000000, 1.1123679,
            -0.0674207, 0.2000000, 1.1123679, -0.0674207, 0.2000000, 1.1123679,
            -0.0674207, 0.2000000, 1.1123679, -0.0674207, 0.2000000, 1.1123679,
            -0.0674207, 0.2000000, 1.1123679, -0.0674207, 0.2000000, 1.1123679},
        Shape{3}, vector<float>{0.3, 0.2, 0.5});
    test_layernorm(
        Shape{2, 3, 2, 3},
        vector<float>{0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,
                      9.,  10., 11., 12., 13., 14., 15., 16., 17.,
                      18., 19., 20., 21., 22., 23., 24., 25., 26.,
                      27., 28., 29., 30., 31., 32., 33., 34., 35.},
        Shape{1}, vector<float>{0.3}, 1e-5, 3, 1,
        vector<float>{
            -0.0674207, 0.2000000, 0.8674207, -0.0674207, 0.2000000, 0.8674207,
            -0.0674207, 0.2000000, 0.8674207, -0.0674207, 0.2000000, 0.8674207,
            -0.0674207, 0.2000000, 0.8674207, -0.0674207, 0.2000000, 0.8674207,
            -0.0674207, 0.2000000, 0.8674207, -0.0674207, 0.2000000, 0.8674207,
            -0.0674207, 0.2000000, 0.8674207, -0.0674207, 0.2000000, 0.8674207,
            -0.0674207, 0.2000000, 0.8674207, -0.0674207, 0.2000000, 0.8674207},
        Shape{3}, vector<float>{0.3, 0.2, 0.5});
    test_layernorm(
        Shape{2, 3, 2, 3},
        vector<float>{0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,
                      9.,  10., 11., 12., 13., 14., 15., 16., 17.,
                      18., 19., 20., 21., 22., 23., 24., 25., 26.,
                      27., 28., 29., 30., 31., 32., 33., 34., 35.},
        Shape{3}, vector<float>{0.3, 0.2, 0.5}, 1e-5, 3, 1,
        vector<float>{-0.3674207, 0.0000000,  0.6123678,  -0.3674207,
                      0.0000000,  0.6123678,  -0.3674207, 0.0000000,
                      0.6123678,  -0.3674207, 0.0000000,  0.6123678,
                      -0.3674207, 0.0000000,  0.6123678,  -0.3674207,
                      0.0000000,  0.6123678,  -0.3674207, 0.0000000,
                      0.6123678,  -0.3674207, 0.0000000,  0.6123678,
                      -0.3674207, 0.0000000,  0.6123678,  -0.3674207,
                      0.0000000,  0.6123678,  -0.3674207, 0.0000000,
                      0.6123678,  -0.3674207, 0.0000000,  0.6123678});

} // python output

} // namespace infini
