#include "cmath"
#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/resize.h"
#include "test.h"
namespace infini {
TEST(Resize, Cuda_downsample_sizes_nearest) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 2, 4}, DataType::Float32);
    auto sizes = gCpu->addTensor({4}, DataType::UInt32);
    gCpu->dataMalloc();
    input->copyData(vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    sizes->copyData(vector<uint32_t>{1, 1, 1, 3});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(
        gCuda->cloneTensor(input), nullptr, std::nullopt,
        gCuda->cloneTensor(sizes), ResizeObj::EKeepAspectRatioPolicy::stretch);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(oCpu->equalData(vector<float>{1, 2, 4}));
}

TEST(Resize, Cuda_upsample_sizes_nearest_notlarger) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 2, 2}, DataType::Float32);
    auto sizes = gCpu->addTensor({2}, DataType::UInt32);
    gCpu->dataMalloc();
    input->copyData(vector<float>{1, 2, 3, 4});
    sizes->copyData(vector<uint32_t>{7, 8});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(
        gCuda->cloneTensor(input), nullptr, vector<int>{2, 3},
        gCuda->cloneTensor(sizes), ResizeObj::EKeepAspectRatioPolicy::notLarger,
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
    Runtime runtime = CpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 2, 2}, DataType::Float32);
    auto sizes = gCpu->addTensor({2}, DataType::UInt32);
    gCpu->dataMalloc();
    input->copyData(vector<float>{1, 2, 3, 4});
    sizes->copyData(vector<uint32_t>{7, 8});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op =
        gCuda->addOp<ResizeObj>(gCuda->cloneTensor(input), nullptr,
                                vector<int>{2, 3}, gCuda->cloneTensor(sizes),
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
    Runtime runtime = CpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 4, 4}, DataType::Float32);
    auto sizes = gCpu->addTensor({4}, DataType::UInt32);
    gCpu->dataMalloc();
    input->copyData(
        vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    sizes->copyData(vector<uint32_t>{1, 1, 8, 8});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(
        gCuda->cloneTensor(input), nullptr, std::nullopt,
        gCuda->cloneTensor(sizes), ResizeObj::EKeepAspectRatioPolicy::stretch,
        ResizeObj::ENearestMode::ceil,
        ResizeObj::ECoordinateTransMode::halfPixel);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    auto o = op->getOutput(0);
    // //cudaPrintTensor(o);
    auto oCpu = gCpu->cloneTensor(o);
    EXPECT_TRUE(oCpu->equalData(vector<float>{
        1,  2,  2,  3,  3,  4,  4,  4,  5,  6,  6,  7,  7,  8,  8,  8,
        5,  6,  6,  7,  7,  8,  8,  8,  9,  10, 10, 11, 11, 12, 12, 12,
        9,  10, 10, 11, 11, 12, 12, 12, 13, 14, 14, 15, 15, 16, 16, 16,
        13, 14, 14, 15, 15, 16, 16, 16, 13, 14, 14, 15, 15, 16, 16, 16}));
}

TEST(Resize, Cuda_upsample_sizes_nearest_floor_align_corners) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 4, 4}, DataType::Float32);
    auto sizes = gCpu->addTensor({2}, DataType::UInt32);
    gCpu->dataMalloc();
    input->copyData(
        vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    sizes->copyData(vector<uint32_t>{8, 8});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(
        gCuda->cloneTensor(input), nullptr, vector<int>{3, 2},
        gCuda->cloneTensor(sizes), ResizeObj::EKeepAspectRatioPolicy::stretch,
        ResizeObj::ENearestMode::floor,
        ResizeObj::ECoordinateTransMode::alignCorners);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    auto o = op->getOutput(0);
    // cudaPrintTensor(o);
    auto oCpu = gCpu->cloneTensor(o);
    EXPECT_TRUE(oCpu->equalData(vector<float>{
        1, 1, 1, 2,  2,  3,  3,  4,  1,  1,  1,  2,  2,  3,  3,  4,
        1, 1, 1, 2,  2,  3,  3,  4,  5,  5,  5,  6,  6,  7,  7,  8,
        5, 5, 5, 6,  6,  7,  7,  8,  9,  9,  9,  10, 10, 11, 11, 12,
        9, 9, 9, 10, 10, 11, 11, 12, 13, 13, 13, 14, 14, 15, 15, 16}));
}

TEST(Resize, Cuda_upsample_sizes_nearest_round_prefer_ceil_asymmetri) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 4, 4}, DataType::Float32);
    auto sizes = gCpu->addTensor({4}, DataType::UInt32);
    gCpu->dataMalloc();
    input->copyData(
        vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    sizes->copyData(vector<uint32_t>{1, 1, 8, 8});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(
        gCuda->cloneTensor(input), nullptr, std::nullopt,
        gCuda->cloneTensor(sizes), ResizeObj::EKeepAspectRatioPolicy::stretch,
        ResizeObj::ENearestMode::roundPreferCeil,
        ResizeObj::ECoordinateTransMode::asymmetric);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    auto o = op->getOutput(0);
    // cudaPrintTensor(o);
    auto oCpu = gCpu->cloneTensor(o);
    EXPECT_TRUE(oCpu->equalData(vector<float>{
        1,  2,  2,  3,  3,  4,  4,  4,  5,  6,  6,  7,  7,  8,  8,  8,
        5,  6,  6,  7,  7,  8,  8,  8,  9,  10, 10, 11, 11, 12, 12, 12,
        9,  10, 10, 11, 11, 12, 12, 12, 13, 14, 14, 15, 15, 16, 16, 16,
        13, 14, 14, 15, 15, 16, 16, 16, 13, 14, 14, 15, 15, 16, 16, 16}));
}

TEST(Resize, Cuda_downsample_scales_nearest) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 2, 4}, DataType::Float32);
    auto scales = gCpu->addTensor({4}, DataType::Float32);
    gCpu->dataMalloc();
    input->copyData(vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    scales->copyData(vector<float>{1, 1, 0.6, 0.6});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(gCuda->cloneTensor(input), nullptr,
                                      std::nullopt, gCuda->cloneTensor(scales));
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(oCpu->equalData(vector<float>{1, 3}));
}

TEST(Resize, Cuda_upsample_scales_nearest) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 2, 2}, DataType::Float32);
    auto scales = gCpu->addTensor({4}, DataType::Float32);
    gCpu->dataMalloc();
    input->copyData(vector<float>{1, 2, 3, 4});
    scales->copyData(vector<float>{1, 1, 2, 3});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(gCuda->cloneTensor(input), nullptr,
                                      std::nullopt, gCuda->cloneTensor(scales));
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(
        oCpu->equalData(vector<float>{1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2,
                                      3, 3, 3, 4, 4, 4, 3, 3, 3, 4, 4, 4}));
}

TEST(Resize, Cuda_upsample_scales_nearest_axes_3_2) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 2, 2}, DataType::Float32);
    auto scales = gCpu->addTensor({2}, DataType::Float32);
    gCpu->dataMalloc();
    input->copyData(vector<float>{1, 2, 3, 4});
    scales->copyData(vector<float>{3, 2});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op =
        gCuda->addOp<ResizeObj>(gCuda->cloneTensor(input), nullptr,
                                vector<int>{3, 2}, gCuda->cloneTensor(scales));
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(
        oCpu->equalData(vector<float>{1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2,
                                      3, 3, 3, 4, 4, 4, 3, 3, 3, 4, 4, 4}));
}

TEST(Resize, Cuda_downsample_scales_linear) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 2, 4}, DataType::Float32);
    auto scales = gCpu->addTensor({4}, DataType::Float32);
    gCpu->dataMalloc();
    input->copyData(vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    scales->copyData(vector<float>{1, 1, 0.6, 0.6});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(gCuda->cloneTensor(input), nullptr,
                                      std::nullopt, gCuda->cloneTensor(scales),
                                      ResizeObj::ECoeffMode::linear);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(oCpu->equalData(vector<float>{2.6666665, 4.3333331}));
}

TEST(Resize, Cuda_upsample_scales_linear) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 2, 2}, DataType::Float32);
    auto scales = gCpu->addTensor({4}, DataType::Float32);
    gCpu->dataMalloc();
    input->copyData(vector<float>{1, 2, 3, 4});
    scales->copyData(vector<float>{1, 1, 2, 2});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(gCuda->cloneTensor(input), nullptr,
                                      std::nullopt, gCuda->cloneTensor(scales),
                                      ResizeObj::ECoeffMode::linear);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(
        oCpu->equalData(vector<float>{1, 1.25, 1.75, 2, 1.5, 1.75, 2.25, 2.5,
                                      2.5, 2.75, 3.25, 3.5, 3, 3.25, 3.75, 4}));
}

TEST(Resize, Cuda_upsample_scales_linear_align_corners) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 2, 2}, DataType::Float32);
    auto scales = gCpu->addTensor({4}, DataType::Float32);
    gCpu->dataMalloc();
    input->copyData(vector<float>{1, 2, 3, 4});
    scales->copyData(vector<float>{1, 1, 2, 2});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(
        gCuda->cloneTensor(input), nullptr, std::nullopt,
        gCuda->cloneTensor(scales), ResizeObj::ECoeffMode::linear,
        ResizeObj::ECoordinateTransMode::alignCorners);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);
    cudaPrintTensor(op->getOutput(0));
    //  copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(oCpu->equalData(vector<float>{
        1, 1.333333, 1.666667, 2, 1.666667, 2, 2.333333, 2.666667, 2.333333,
        2.6666667, 3, 3.333333, 3, 3.333333, 3.6666667, 4}));
}

TEST(Resize, Cuda_downsample_sizes_linear_pytorchhalfpixel) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({1, 1, 4, 4}, DataType::Float32);
    auto sizes = gCpu->addTensor({4}, DataType::UInt32);
    gCpu->dataMalloc();
    input->copyData(
        vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    sizes->copyData(vector<uint32_t>{1, 1, 3, 1});

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op = gCuda->addOp<ResizeObj>(
        gCuda->cloneTensor(input), nullptr, std::nullopt,
        gCuda->cloneTensor(sizes), ResizeObj::EKeepAspectRatioPolicy::stretch,
        ResizeObj::ECoeffMode::linear,
        ResizeObj::ECoordinateTransMode::pytorchHalfPixel);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    auto oCpu = gCpu->cloneTensor(op->getOutput(0));
    // cudaPrintTensor(op->getOutput(0));
    EXPECT_TRUE(oCpu->equalData(vector<float>{1.666667, 7, 12.33333}));
}

} // namespace infini
