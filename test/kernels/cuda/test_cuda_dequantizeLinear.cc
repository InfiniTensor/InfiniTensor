#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/dequantizeLinear.h"

#include "test.h"

namespace infini {

void test_dequantizeLinearFp32(
    const Shape &inputXShape, const vector<uint8_t> &inputXData,
    const Shape &inputScaleShape, const vector<float> &inputScaleData, int axis,
    const vector<float> &ExpectData,
    const std::optional<Shape> &zeroPointShape = std::nullopt,
    const std::optional<std::vector<uint8_t>> &inputZeroPointData =
        std::nullopt) {

    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    if (zeroPointShape.has_value() && inputZeroPointData.has_value()) {
        Shape inputZeroPointShape = *zeroPointShape;

        auto inputZeroPoint =
            gCpu->addTensor(inputZeroPointShape, DataType::UInt8);
        auto inputX = gCpu->addTensor(inputXShape, DataType::UInt8);
        auto inputScale = gCpu->addTensor(inputScaleShape, DataType::Float32);
        gCpu->dataMalloc();
        inputZeroPoint->copyin(*inputZeroPointData); //
        inputX->copyin(inputXData);
        inputScale->copyin(inputScaleData); //

        // inputX->printData();
        // inputZeroPoint->printData();
        // inputScale->printData();

        auto cudaRuntime = make_ref<CudaRuntimeObj>();
        Graph gCuda = make_ref<GraphObj>(cudaRuntime);
        auto inputZeroPointGpu = gCuda->cloneTensor(inputZeroPoint);
        auto inputXGpu = gCuda->cloneTensor(inputX);
        auto inputScaleGpu = gCuda->cloneTensor(inputScale);

        auto op = gCuda->addOp<DequantizeLinearObj>(
            inputXGpu, inputScaleGpu, nullptr, inputZeroPointGpu,
            axis); // DequantizeLinearObj
        gCuda->dataMalloc();
        inputZeroPointGpu->copyin(*inputZeroPointData);
        // gCpu->cloneTensor(inputZeroPointGpu)->printData();
        inputXGpu->copyin(inputXData);
        inputScaleGpu->copyin(inputScaleData);
        cudaRuntime->run(gCuda);

        auto oCpu =
            gCpu->cloneTensor(op->getOutput()); // move Data from gpu to cpu
        oCpu->printData();                      //->printData
        EXPECT_TRUE(oCpu->equalData(ExpectData));
    } else {

        auto inputX = gCpu->addTensor(inputXShape, DataType::UInt8);
        auto inputScale = gCpu->addTensor(inputScaleShape, DataType::Float32);
        gCpu->dataMalloc();

        inputX->copyin(inputXData);
        inputScale->copyin(inputScaleData); //
        auto cudaRuntime = make_ref<CudaRuntimeObj>();
        Graph gCuda = make_ref<GraphObj>(cudaRuntime);

        auto inputXGpu = gCuda->cloneTensor(inputX);
        auto inputScaleGpu = gCuda->cloneTensor(inputScale);
        auto op = gCuda->addOp<DequantizeLinearObj>(
            inputXGpu, inputScaleGpu, nullptr, nullptr,
            axis); // DequantizeLinearObj
        gCuda->dataMalloc();

        inputXGpu->copyin(inputXData);
        inputScaleGpu->copyin(inputScaleData);
        cudaRuntime->run(gCuda);

        auto oCpu =
            gCpu->cloneTensor(op->getOutput()); // move Data from gpu to cpu
        oCpu->printData();                      //->printData
        EXPECT_TRUE(oCpu->equalData(ExpectData));
    }
}
void test_dequantizeLinearFp16(
    const Shape &inputXShape, const vector<uint8_t> &inputXData,
    const Shape &inputScaleShape,
    const std::function<void(void *, size_t, DataType)> &generator, int axis,
    const vector<float> &ExpectData,
    const std::optional<Shape> &zeroPointShape = std::nullopt,
    const std::optional<std::vector<uint8_t>> &inputZeroPointData =
        std::nullopt) {

    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    if (zeroPointShape.has_value() && inputZeroPointData.has_value()) {
        Shape inputZeroPointShape = *zeroPointShape;

        auto inputZeroPoint =
            gCpu->addTensor(inputZeroPointShape, DataType::UInt8);
        auto inputX = gCpu->addTensor(inputXShape, DataType::UInt8);
        auto inputScale = gCpu->addTensor(inputScaleShape, DataType::Float16);
        gCpu->dataMalloc();
        inputZeroPoint->copyin(*inputZeroPointData); //
        // inputZeroPoint->printData();
        inputX->copyin(inputXData);
        inputScale->setData(generator);
        auto cudaRuntime = make_ref<CudaRuntimeObj>();
        Graph gCuda = make_ref<GraphObj>(cudaRuntime);
        auto inputZeroPointGpu = gCuda->cloneTensor(inputZeroPoint);
        auto inputXGpu = gCuda->cloneTensor(inputX);
        auto inputScaleGpu = gCuda->cloneTensor(inputScale);
        // gCpu->cloneTensor(inputZeroPointGpu)->printData();
        auto op = gCuda->addOp<DequantizeLinearObj>(
            inputXGpu, inputScaleGpu, nullptr, inputZeroPointGpu,
            axis); // DequantizeLinearObj
        gCuda->dataMalloc();
        inputZeroPointGpu->copyin(*inputZeroPointData);
        // gCpu->cloneTensor(inputZeroPointGpu)->printData();
        inputXGpu->copyin(inputXData);
        inputScaleGpu->setData(generator);
        cudaRuntime->run(gCuda);

        auto oCpu =
            gCpu->cloneTensor(op->getOutput()); // move Data from gpu to cpu
        oCpu->printData();                      //->printData
        EXPECT_TRUE(oCpu->equalData(ExpectData));
    } else {

        auto inputX = gCpu->addTensor(inputXShape, DataType::UInt8);
        auto inputScale = gCpu->addTensor(inputScaleShape, DataType::Float16);
        gCpu->dataMalloc();

        inputX->copyin(inputXData);
        inputScale->setData(generator);
        auto cudaRuntime = make_ref<CudaRuntimeObj>();
        Graph gCuda = make_ref<GraphObj>(cudaRuntime);

        auto inputXGpu = gCuda->cloneTensor(inputX);
        auto inputScaleGpu = gCuda->cloneTensor(inputScale);
        auto op = gCuda->addOp<DequantizeLinearObj>(
            inputXGpu, inputScaleGpu, nullptr, nullptr,
            axis); // DequantizeLinearObj
        gCuda->dataMalloc();

        inputXGpu->copyin(inputXData);
        inputScaleGpu->setData(generator);
        cudaRuntime->run(gCuda);

        auto oCpu =
            gCpu->cloneTensor(op->getOutput()); // move Data from gpu to cpu
        oCpu->printData();                      //->printData
        EXPECT_TRUE(oCpu->equalData(ExpectData));
    }
}

TEST(CUDA_DequantizeLinearFp32, run) {

    test_dequantizeLinearFp32(
        Shape{2, 3, 2, 3},
        vector<uint8_t>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35},
        Shape{3}, vector<float>{0.3, 0.2, 0.5}, 1,
        vector<float>{-0.3000000, 0.0000000,  0.3000000,  0.6000000,
                      0.9000000,  1.2000000,  0.8000000,  1.0000000,
                      1.2000000,  1.4000000,  1.6000000,  1.8000001,
                      4.5000000,  5.0000000,  5.5000000,  6.0000000,
                      6.5000000,  7.0000000,  5.1000004,  5.4000001,
                      5.7000003,  6.0000000,  6.3000002,  6.6000004,
                      4.4000001,  4.5999999,  4.8000002,  5.0000000,
                      5.2000003,  5.4000001,  13.5000000, 14.0000000,
                      14.5000000, 15.0000000, 15.5000000, 16.0000000},
        Shape{3}, vector<uint8_t>{1, 2, 3});
    test_dequantizeLinearFp32(
        Shape{2, 3, 2, 3},
        vector<uint8_t>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35},
        Shape{3}, vector<float>{0.3, 0.2, 0.5}, 1,
        vector<float>{0.0000000,  0.3000000,  0.6000000,  0.9000000,
                      1.2000000,  1.5000000,  1.2000000,  1.4000000,
                      1.6000000,  1.8000001,  2.0000000,  2.2000000,
                      6.0000000,  6.5000000,  7.0000000,  7.5000000,
                      8.0000000,  8.5000000,  5.4000001,  5.7000003,
                      6.0000000,  6.3000002,  6.6000004,  6.9000001,
                      4.8000002,  5.0000000,  5.2000003,  5.4000001,
                      5.5999999,  5.8000002,  15.0000000, 15.5000000,
                      16.0000000, 16.5000000, 17.0000000, 17.5000000});

} // python output
TEST(CUDA_DequantizeLinearFp16, run) {
    test_dequantizeLinearFp16(
        Shape{2, 3, 2, 3},
        vector<uint8_t>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35},
        Shape{3}, ValGenerator<2>(), 1,
        vector<float>{-2., 0.,  2.,  4.,  6.,  8.,  8.,  10., 12.,
                      14., 16., 18., 18., 20., 22., 24., 26., 28.,
                      34., 36., 38., 40., 42., 44., 44., 46., 48.,
                      50., 52., 54., 54., 56., 58., 60., 62., 64.},
        Shape{3}, vector<uint8_t>{1, 2, 3});
    test_dequantizeLinearFp16(
        Shape{2, 3, 2, 3},
        vector<uint8_t>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35},
        Shape{3}, ValGenerator<2>(), 1,
        vector<float>{0.,  2.,  4.,  6.,  8.,  10., 12., 14., 16.,
                      18., 20., 22., 24., 26., 28., 30., 32., 34.,
                      36., 38., 40., 42., 44., 46., 48., 50., 52.,
                      54., 56., 58., 60., 62., 64., 66., 68., 70.});

} // python output

} // namespace infini
