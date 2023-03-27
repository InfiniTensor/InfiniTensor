#include "cmath"
#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/resize.h"
#include "test.h"
namespace infini {
TEST(Resize, Cuda_downsample_sizes_nearest) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 2, 4}, DataType::Float32);
    auto sizes = gCpu->addTensor({4}, DataType::UInt32);
    gCpu->dataMalloc();
    input->copyin(vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    sizes->copyin(vector<uint32_t>{1, 1, 1, 3});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(
        gCuda->cloneTensor(input), nullptr, std::nullopt,
        gCuda->cloneTensor(sizes), nullptr, nullptr,
        ResizeObj::EKeepAspectRatioPolicy::stretch);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(oCpu->equalData(vector<float>{1, 2, 4}));
}

TEST(Resize, Cuda_upsample_sizes_nearest_notlarger) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 2, 2}, DataType::Float32);
    auto sizes = gCpu->addTensor({2}, DataType::UInt32);
    gCpu->dataMalloc();
    input->copyin(vector<float>{1, 2, 3, 4});
    sizes->copyin(vector<uint32_t>{7, 8});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(
        gCuda->cloneTensor(input), nullptr, vector<int>{2, 3},
        gCuda->cloneTensor(sizes), nullptr, nullptr,
        ResizeObj::EKeepAspectRatioPolicy::notLarger,
        ResizeObj::ENearestMode::roundPreferFloor,
        ResizeObj::ECoordinateTransMode::halfPixel);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(oCpu->equalData(
        vector<float>{1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1,
                      1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4,
                      4, 3, 3, 3, 3, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4}));
}

TEST(Resize, Cuda_upsample_sizes_nearest_notsmaller) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 2, 2}, DataType::Float32);
    auto sizes = gCpu->addTensor({2}, DataType::UInt32);
    gCpu->dataMalloc();
    input->copyin(vector<float>{1, 2, 3, 4});
    sizes->copyin(vector<uint32_t>{7, 8});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(
        gCuda->cloneTensor(input), nullptr, vector<int>{2, 3},
        gCuda->cloneTensor(sizes), nullptr, nullptr,
        ResizeObj::EKeepAspectRatioPolicy::notSmaller,
        ResizeObj::ENearestMode::roundPreferFloor,
        ResizeObj::ECoordinateTransMode::halfPixel);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(oCpu->equalData(vector<float>{
        1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2,
        2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3,
        4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4}));
}

TEST(Resize, Cuda_upsample_sizes_nearest_ceil_half_pixel) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 4, 4}, DataType::Float32);
    auto sizes = gCpu->addTensor({4}, DataType::UInt32);
    gCpu->dataMalloc();
    input->copyin(
        vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    sizes->copyin(vector<uint32_t>{1, 1, 8, 8});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(
        gCuda->cloneTensor(input), nullptr, std::nullopt,
        gCuda->cloneTensor(sizes), nullptr, nullptr,
        ResizeObj::EKeepAspectRatioPolicy::stretch,
        ResizeObj::ENearestMode::ceil,
        ResizeObj::ECoordinateTransMode::halfPixel);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    auto o = op->getOutput(0);
    auto oCpu = gCpu->cloneTensor(o);
    EXPECT_TRUE(oCpu->equalData(vector<float>{
        1,  2,  2,  3,  3,  4,  4,  4,  5,  6,  6,  7,  7,  8,  8,  8,
        5,  6,  6,  7,  7,  8,  8,  8,  9,  10, 10, 11, 11, 12, 12, 12,
        9,  10, 10, 11, 11, 12, 12, 12, 13, 14, 14, 15, 15, 16, 16, 16,
        13, 14, 14, 15, 15, 16, 16, 16, 13, 14, 14, 15, 15, 16, 16, 16}));
}

TEST(Resize, Cuda_upsample_sizes_nearest_floor_align_corners) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 4, 4}, DataType::Float32);
    auto sizes = gCpu->addTensor({2}, DataType::UInt32);
    gCpu->dataMalloc();
    input->copyin(
        vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    sizes->copyin(vector<uint32_t>{8, 8});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(
        gCuda->cloneTensor(input), nullptr, vector<int>{3, 2},
        gCuda->cloneTensor(sizes), nullptr, nullptr,
        ResizeObj::EKeepAspectRatioPolicy::stretch,
        ResizeObj::ENearestMode::floor,
        ResizeObj::ECoordinateTransMode::alignCorners);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    auto o = op->getOutput(0);
    auto oCpu = gCpu->cloneTensor(o);
    EXPECT_TRUE(oCpu->equalData(vector<float>{
        1, 1, 1, 2,  2,  3,  3,  4,  1,  1,  1,  2,  2,  3,  3,  4,
        1, 1, 1, 2,  2,  3,  3,  4,  5,  5,  5,  6,  6,  7,  7,  8,
        5, 5, 5, 6,  6,  7,  7,  8,  9,  9,  9,  10, 10, 11, 11, 12,
        9, 9, 9, 10, 10, 11, 11, 12, 13, 13, 13, 14, 14, 15, 15, 16}));
}

TEST(Resize, Cuda_upsample_sizes_nearest_round_prefer_ceil_asymmetri) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 4, 4}, DataType::Float32);
    auto sizes = gCpu->addTensor({4}, DataType::UInt32);
    gCpu->dataMalloc();
    input->copyin(
        vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    sizes->copyin(vector<uint32_t>{1, 1, 8, 8});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(
        gCuda->cloneTensor(input), nullptr, std::nullopt,
        gCuda->cloneTensor(sizes), nullptr, nullptr,
        ResizeObj::EKeepAspectRatioPolicy::stretch,
        ResizeObj::ENearestMode::roundPreferCeil,
        ResizeObj::ECoordinateTransMode::asymmetric);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    auto o = op->getOutput(0);
    auto oCpu = gCpu->cloneTensor(o);
    EXPECT_TRUE(oCpu->equalData(vector<float>{
        1,  2,  2,  3,  3,  4,  4,  4,  5,  6,  6,  7,  7,  8,  8,  8,
        5,  6,  6,  7,  7,  8,  8,  8,  9,  10, 10, 11, 11, 12, 12, 12,
        9,  10, 10, 11, 11, 12, 12, 12, 13, 14, 14, 15, 15, 16, 16, 16,
        13, 14, 14, 15, 15, 16, 16, 16, 13, 14, 14, 15, 15, 16, 16, 16}));
}

TEST(Resize, Cuda_downsample_scales_nearest) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 2, 4}, DataType::Float32);
    auto scales = gCpu->addTensor({4}, DataType::Float32);
    gCpu->dataMalloc();
    input->copyin(vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    scales->copyin(vector<float>{1, 1, 0.6, 0.6});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(gCuda->cloneTensor(input), nullptr,
                                      std::nullopt, nullptr,
                                      gCuda->cloneTensor(scales), nullptr);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(oCpu->equalData(vector<float>{1, 3}));
}

TEST(Resize, Cuda_upsample_scales_nearest) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 2, 2}, DataType::Float32);
    auto scales = gCpu->addTensor({4}, DataType::Float32);
    gCpu->dataMalloc();
    input->copyin(vector<float>{1, 2, 3, 4});
    scales->copyin(vector<float>{1, 1, 2, 3});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(gCuda->cloneTensor(input), nullptr,
                                      std::nullopt, nullptr,
                                      gCuda->cloneTensor(scales), nullptr);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(
        oCpu->equalData(vector<float>{1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2,
                                      3, 3, 3, 4, 4, 4, 3, 3, 3, 4, 4, 4}));
}

TEST(Resize, Cuda_upsample_scales_nearest_axes_3_2) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 2, 2}, DataType::Float32);
    auto scales = gCpu->addTensor({2}, DataType::Float32);
    gCpu->dataMalloc();
    input->copyin(vector<float>{1, 2, 3, 4});
    scales->copyin(vector<float>{3, 2});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(gCuda->cloneTensor(input), nullptr,
                                      vector<int>{3, 2}, nullptr,
                                      gCuda->cloneTensor(scales), nullptr);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(
        oCpu->equalData(vector<float>{1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2,
                                      3, 3, 3, 4, 4, 4, 3, 3, 3, 4, 4, 4}));
}

TEST(Resize, Cuda_downsample_scales_linear) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 2, 4}, DataType::Float32);
    auto scales = gCpu->addTensor({4}, DataType::Float32);
    gCpu->dataMalloc();
    input->copyin(vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    scales->copyin(vector<float>{1, 1, 0.6, 0.6});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(
        gCuda->cloneTensor(input), nullptr, std::nullopt, nullptr,
        gCuda->cloneTensor(scales), nullptr, ResizeObj::ECoeffMode::linear);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(oCpu->equalData(vector<float>{2.6666665, 4.3333331}));
}

TEST(Resize, Cuda_downsample_scales_linear_aligncorners) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 2, 4}, DataType::Float32);
    auto scales = gCpu->addTensor({4}, DataType::Float32);
    gCpu->dataMalloc();
    input->copyin(vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    scales->copyin(vector<float>{1, 1, 0.6, 0.6});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(
        gCuda->cloneTensor(input), nullptr, std::nullopt, nullptr,
        gCuda->cloneTensor(scales), nullptr, ResizeObj::ECoeffMode::linear,
        ResizeObj::EKeepAspectRatioPolicy::none,
        ResizeObj::ECoordinateTransMode::alignCorners);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(oCpu->equalData(vector<float>{1, 3.142857}));
}

TEST(Resize, Cuda_upsample_scales_linear) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 2, 2}, DataType::Float32);
    auto scales = gCpu->addTensor({4}, DataType::Float32);
    gCpu->dataMalloc();
    input->copyin(vector<float>{1, 2, 3, 4});
    scales->copyin(vector<float>{1, 1, 2, 2});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(
        gCuda->cloneTensor(input), nullptr, std::nullopt, nullptr,
        gCuda->cloneTensor(scales), nullptr, ResizeObj::ECoeffMode::linear);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(
        oCpu->equalData(vector<float>{1, 1.25, 1.75, 2, 1.5, 1.75, 2.25, 2.5,
                                      2.5, 2.75, 3.25, 3.5, 3, 3.25, 3.75, 4}));
}

TEST(Resize, Cuda_upsample_scales_linear_align_corners) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 2, 2}, DataType::Float32);
    auto scales = gCpu->addTensor({4}, DataType::Float32);
    gCpu->dataMalloc();
    input->copyin(vector<float>{1, 2, 3, 4});
    scales->copyin(vector<float>{1, 1, 2, 2});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(
        gCuda->cloneTensor(input), nullptr, std::nullopt, nullptr,
        gCuda->cloneTensor(scales), nullptr, ResizeObj::ECoeffMode::linear,
        ResizeObj::EKeepAspectRatioPolicy::none,
        ResizeObj::ECoordinateTransMode::alignCorners);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(oCpu->equalData(vector<float>{
        1, 1.333333, 1.666667, 2, 1.666667, 2, 2.333333, 2.666667, 2.333333,
        2.6666667, 3, 3.333333, 3, 3.333333, 3.6666667, 4}));
}

TEST(Resize, Cuda_downsample_sizes_linear_pytorchhalfpixel) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 4, 4}, DataType::Float32);
    auto sizes = gCpu->addTensor({4}, DataType::UInt32);
    gCpu->dataMalloc();
    input->copyin(
        vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    sizes->copyin(vector<uint32_t>{1, 1, 3, 1});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(
        gCuda->cloneTensor(input), nullptr, std::nullopt,
        gCuda->cloneTensor(sizes), nullptr, nullptr,
        ResizeObj::ECoeffMode::linear,
        ResizeObj::EKeepAspectRatioPolicy::stretch,
        ResizeObj::ECoordinateTransMode::pytorchHalfPixel);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(oCpu->equalData(vector<float>{1.666667, 7, 12.33333}));
}

TEST(Resize, Cuda_tf_crop_and_resize) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 4, 4}, DataType::Float32);
    auto sizes = gCpu->addTensor({4}, DataType::UInt32);
    auto roi = gCpu->addTensor({8}, DataType::Float32);
    gCpu->dataMalloc();
    input->copyin(
        vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    sizes->copyin(vector<uint32_t>{1, 1, 3, 3});
    roi->copyin(vector<float>{0, 0, 0.4, 0.6, 1, 1, 0.6, 0.8});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(
        gCuda->cloneTensor(input), nullptr, std::nullopt,
        gCuda->cloneTensor(sizes), nullptr, gCuda->cloneTensor(roi),
        ResizeObj::ECoeffMode::linear,
        ResizeObj::EKeepAspectRatioPolicy::stretch,
        ResizeObj::ECoordinateTransMode::tfCropAndResize);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(oCpu->equalData(vector<float>{7.6000004, 7.9, 8.2, 8.8, 9.1,
                                              9.400001, 10, 10.3, 10.6}));
}

TEST(Resize, Cuda_tf_crop_and_resize_axes_3_2) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 4, 4}, DataType::Float32);
    auto sizes = gCpu->addTensor({2}, DataType::UInt32);
    auto roi = gCpu->addTensor({4}, DataType::Float32);
    gCpu->dataMalloc();
    input->copyin(
        vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    sizes->copyin(vector<uint32_t>{3, 3});
    roi->copyin(vector<float>{0.6, 0.4, 0.8, 0.6});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(
        gCuda->cloneTensor(input), nullptr, vector<int>{3, 2},
        gCuda->cloneTensor(sizes), nullptr, gCuda->cloneTensor(roi),
        ResizeObj::ECoeffMode::linear,
        ResizeObj::EKeepAspectRatioPolicy::stretch,
        ResizeObj::ECoordinateTransMode::tfCropAndResize);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(oCpu->equalData(vector<float>{7.6000004, 7.9, 8.2, 8.8, 9.1,
                                              9.400001, 10, 10.3, 10.6}));
}

TEST(Resize, Cuda_downsample_scales_cubic) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 4, 4}, DataType::Float32);
    auto scales = gCpu->addTensor({4}, DataType::Float32);
    gCpu->dataMalloc();
    input->copyin(
        vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    scales->copyin(vector<float>{1.0, 1.0, 0.8, 0.8});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(
        gCuda->cloneTensor(input), nullptr, std::nullopt, nullptr,
        gCuda->cloneTensor(scales), nullptr, ResizeObj::ECoeffMode::cubic);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    //   copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(oCpu->equalData(
        vector<float>{1.47119141, 2.78125, 4.08251953, 6.71142578, 8.02148438,
                      9.32275391, 11.91650391, 13.2265625, 14.52783203}));
}

TEST(Resize, Cuda_downsample_scales_cubic_align_corners) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 4, 4}, DataType::Float32);
    auto scales = gCpu->addTensor({4}, DataType::Float32);
    gCpu->dataMalloc();
    input->copyin(
        vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    scales->copyin(vector<float>{1.0, 1.0, 0.8, 0.8});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(
        gCuda->cloneTensor(input), nullptr, std::nullopt, nullptr,
        gCuda->cloneTensor(scales), nullptr, ResizeObj::ECoeffMode::cubic,
        ResizeObj::EKeepAspectRatioPolicy::none,
        ResizeObj::ECoordinateTransMode::alignCorners);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);
    //   copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(oCpu->equalData(
        vector<float>{1, 2.39519159, 3.79038317, 6.58076634, 7.97595793,
                      9.37114951, 12.16153268, 13.55672427, 14.95191585}));
}

TEST(Resize, Cuda_upsample_scales_cubic) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 4, 4}, DataType::Float32);
    auto scales = gCpu->addTensor({4}, DataType::Float32);
    gCpu->dataMalloc();
    input->copyin(
        vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    scales->copyin(vector<float>{1.0, 1.0, 2, 2});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(
        gCuda->cloneTensor(input), nullptr, std::nullopt, nullptr,
        gCuda->cloneTensor(scales), nullptr, ResizeObj::ECoeffMode::cubic);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);
    //   copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(oCpu->equalData(vector<float>{
        0.47265625,  0.76953125,  1.24609375,  1.875,       2.28125,
        2.91015625,  3.38671875,  3.68359375,  1.66015625,  1.95703125,
        2.43359375,  3.0625,      3.46875,     4.09765625,  4.57421875,
        4.87109375,  3.56640625,  3.86328125,  4.33984375,  4.96875,
        5.375,       6.00390625,  6.48046875,  6.77734375,  6.08203125,
        6.37890625,  6.85546875,  7.484375,    7.890625,    8.51953125,
        8.99609375,  9.29296875,  7.70703125,  8.00390625,  8.48046875,
        9.109375,    9.515625,    10.14453125, 10.62109375, 10.91796875,
        10.22265625, 10.51953125, 10.99609375, 11.625,      12.03125,
        12.66015625, 13.13671875, 13.43359375, 12.12890625, 12.42578125,
        12.90234375, 13.53125,    13.9375,     14.56640625, 15.04296875,
        15.33984375, 13.31640625, 13.61328125, 14.08984375, 14.71875,
        15.125,      15.75390625, 16.23046875, 16.52734375}));
}

TEST(Resize, Cuda_upsample_scales_cubic_align_corners) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 4, 4}, DataType::Float32);
    auto scales = gCpu->addTensor({4}, DataType::Float32);
    gCpu->dataMalloc();
    input->copyin(
        vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    scales->copyin(vector<float>{1.0, 1.0, 2, 2});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(
        gCuda->cloneTensor(input), nullptr, std::nullopt, nullptr,
        gCuda->cloneTensor(scales), nullptr, ResizeObj::ECoeffMode::cubic,
        ResizeObj::EKeepAspectRatioPolicy::none,
        ResizeObj::ECoordinateTransMode::alignCorners);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);
    //   copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(oCpu->equalData(vector<float>{
        1,           1.34110787,  1.80029155,  2.32944606,  2.67055394,
        3.19970845,  3.65889213,  4,           2.36443149,  2.70553936,
        3.16472303,  3.69387755,  4.03498542,  4.56413994,  5.02332362,
        5.36443149,  4.20116618,  4.54227405,  5.00145773,  5.53061224,
        5.87172012,  6.40087464,  6.86005831,  7.20116618,  6.31778426,
        6.65889213,  7.1180758,   7.64723032,  7.98833819,  8.51749271,
        8.97667638,  9.31778426,  7.68221574,  8.02332362,  8.48250729,
        9.01166181,  9.35276968,  9.8819242,   10.34110787, 10.68221574,
        9.79883382,  10.13994169, 10.59912536, 11.12827988, 11.46938776,
        11.99854227, 12.45772595, 12.79883382, 11.63556851, 11.97667638,
        12.43586006, 12.96501458, 13.30612245, 13.83527697, 14.29446064,
        14.63556851, 13,          13.34110787, 13.80029155, 14.32944606,
        14.67055394, 15.19970845, 15.65889213, 16.}));
}

TEST(Resize, Cuda_upsample_scales_cubic_asymmetric) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 4, 4}, DataType::Float32);
    auto scales = gCpu->addTensor({4}, DataType::Float32);
    gCpu->dataMalloc();
    input->copyin(
        vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    scales->copyin(vector<float>{1.0, 1.0, 2, 2});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(
        gCuda->cloneTensor(input), nullptr, std::nullopt, nullptr,
        gCuda->cloneTensor(scales), nullptr, ResizeObj::ECoeffMode::cubic,
        ResizeObj::EKeepAspectRatioPolicy::none,
        ResizeObj::ECoordinateTransMode::asymmetric);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);
    //   copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(oCpu->equalData(vector<float>{
        1.,     1.40625,  2.,     2.5,    3.,     3.59375,  4.,     4.09375,
        2.625,  3.03125,  3.625,  4.125,  4.625,  5.21875,  5.625,  5.71875,
        5.,     5.40625,  6.,     6.5,    7.,     7.59375,  8.,     8.09375,
        7.,     7.40625,  8.,     8.5,    9.,     9.59375,  10.,    10.09375,
        9.,     9.40625,  10.,    10.5,   11.,    11.59375, 12.,    12.09375,
        11.375, 11.78125, 12.375, 12.875, 13.375, 13.96875, 14.375, 14.46875,
        13.,    13.40625, 14.,    14.5,   15.,    15.59375, 16.,    16.09375,
        13.375, 13.78125, 14.375, 14.875, 15.375, 15.96875, 16.375, 16.46875}));
}

//
TEST(Resize, Cuda_downsample_sizes_cubic) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 4, 4}, DataType::Float32);
    auto sizes = gCpu->addTensor({4}, DataType::UInt32);
    gCpu->dataMalloc();
    input->copyin(
        vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    sizes->copyin(vector<uint32_t>{1, 1, 3, 3});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op =
        gCuda->addOp<ResizeObj>(gCuda->cloneTensor(input), nullptr,
                                std::nullopt, gCuda->cloneTensor(sizes),
                                nullptr, nullptr, ResizeObj::ECoeffMode::cubic,
                                ResizeObj::EKeepAspectRatioPolicy::stretch);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    //   copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));

    /*The corresponding test's output of ONNX has some bias, which is:
       {1.63078704, 3.00462963, 4.37847222, 7.12615741, 8.5,
                      9.87384259, 12.62152778, 13.99537037, 15.36921296}
                      (https://github.com/onnx/onnx/blob/main/docs/Operators.md#Resize)*/
    EXPECT_TRUE(oCpu->equalData(vector<float>{
        1.63078511, 3.00462794, 4.37846994, 7.12615490, 8.50000000, 9.87384224,
        12.62152576, 13.99537086, 15.36921501}));
}

TEST(Resize, Cuda_upsample_sizes_cubic) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 4, 4}, DataType::Float32);
    auto sizes = gCpu->addTensor({4}, DataType::UInt32);
    gCpu->dataMalloc();
    input->copyin(
        vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    sizes->copyin(vector<uint32_t>{1, 1, 9, 10});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op =
        gCuda->addOp<ResizeObj>(gCuda->cloneTensor(input), nullptr,
                                std::nullopt, gCuda->cloneTensor(sizes),
                                nullptr, nullptr, ResizeObj::ECoeffMode::cubic,
                                ResizeObj::EKeepAspectRatioPolicy::stretch);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);
    //   copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(oCpu->equalData(vector<float>{
        0.45508048,  0.64058018,  0.97158027,  1.42258000,  1.90733004,
        2.22333097,  2.70807934,  3.15908003,  3.49008012,  3.67558002,
        1.39437866,  1.57987845,  1.91087842,  2.36187792,  2.84662747,
        3.16262865,  3.64737630,  4.09837723,  4.42937851,  4.61487770,
        2.95131063,  3.13681102,  3.46781015,  3.91881013,  4.40356016,
        4.71956062,  5.20430803,  5.65531015,  5.98631001,  6.17181063,
        5.20525312,  5.39075279,  5.72175217,  6.17275286,  6.65750170,
        6.97350359,  7.45825005,  7.90925360,  8.24025249,  8.42575359,
        6.88975096,  7.07525015,  7.40625000,  7.85725021,  8.34200001,
        8.65800095,  9.14274597,  9.59375000,  9.92474842,  10.11025047,
        8.57425022,  8.75974846,  9.09074879,  9.54174805,  10.02649689,
        10.34249973, 10.82724571, 11.27824974, 11.60924721, 11.79474831,
        10.82819176, 11.01369190, 11.34469223, 11.79569244, 12.28044128,
        12.59644127, 13.08118820, 13.53219128, 13.86318874, 14.04869366,
        12.38512325, 12.57062244, 12.90162182, 13.35262108, 13.83737183,
        14.15337372, 14.63811684, 15.08912182, 15.42011929, 15.60562229,
        13.32442474, 13.50992107, 13.84092331, 14.29192352, 14.77667332,
        15.09267426, 15.57741737, 16.02842331, 16.35941887, 16.54491997}));
    /* The corresponding test's output of ONNX has some bias, which is:
    0.45507922,  0.64057922,  0.97157922,  1.42257922,  1.90732922,
    2.22332922,  2.70807922,  3.15907922,  3.49007922,  3.67557922,
    1.39437963,  1.57987963,  1.91087963,  2.36187963,  2.84662963,
    3.16262963,  3.64737963,  4.09837963,  4.42937963,  4.61487963,
    2.95130693,  3.13680693,  3.46780693,  3.91880693,  4.40355693,
    4.71955693,  5.20430693,  5.65530693,  5.98630693,  6.17180693,
    5.20525069,  5.39075069,  5.72175069,  6.17275069,  6.65750069,
    6.97350069,  7.45825069,  7.90925069,  8.24025069,  8.42575069,
    6.88975,     7.07525,     7.40625,     7.85725,     8.342,
    8.658,       9.14275,     9.59375,     9.92475,     10.11025,
    8.57424931,  8.75974931,  9.09074931,  9.54174931,  10.02649931,
    10.34249931, 10.82724931, 11.27824931, 11.60924931, 11.79474931,
    10.82819307, 11.01369307, 11.34469307, 11.79569307, 12.28044307,
    12.59644307, 13.08119307, 13.53219307, 13.86319307, 14.04869307,
    12.38512037, 12.57062037, 12.90162037, 13.35262037, 13.83737037,
    14.15337037, 14.63812037, 15.08912037, 15.42012037, 15.60562037,
    13.32442078, 13.50992078, 13.84092078, 14.29192078, 14.77667078,
    15.09267078, 15.57742078, 16.02842078, 16.35942078, 16.54492078}*/
}

} // namespace infini
