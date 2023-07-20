#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/conv.h"
#include "benchmark.h"
#include <iostream>
#include <cmath>
#include <chrono>
#include <sys/time.h>

using namespace infini;

int main() {

    int warmupRounds = 200;
    int timingRounds = 200;
    DataType dtype = DataType::Float32;

    Shape INPUT_SHAPE = {16, 128, 112, 112};
    Shape KERNEL_SHAPE = {128, 256, 3, 3};
    int NUM_GROUPS = 1;

    int PAD_HEIGHT = 0;
    int PAD_WIDTH = 0;
    int VERTICAL_STRIDE = 1;
    int HORIZONTAL_STRIDE = 1;
    int DILATION_HEIGHT = 1;
    int DILATION_WIDTH = 1;

    size_t inputSize = 1;
    for (auto dim: INPUT_SHAPE) {
        inputSize *= dim;
    }
    size_t inputSizeInBytes = inputSize * sizeof(dtype);

    size_t kernelSize = 1;
    for (auto dim: KERNEL_SHAPE) {
        kernelSize *= dim;
    }
    size_t kernelSizeInBytes = kernelSize * sizeof(dtype);

    // Init time
    double time_memcpy_htod = 0.0, time_memcpy_dtoh = 0.0;
    double time_op = 0.0;

    // Create runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu =
        make_ref<TensorObj>(INPUT_SHAPE, dtype, cpuRuntime);
    inputCpu->dataMalloc();
    inputCpu->setData(RandomGenerator());

    Tensor kernelCpu =
        make_ref<TensorObj>(KERNEL_SHAPE, dtype, cpuRuntime);
    kernelCpu->dataMalloc();
    kernelCpu->setData(RandomGenerator());

    // Build input data on GPU
    Tensor inputGpu = 
        make_ref<TensorObj>(INPUT_SHAPE, dtype, cudaRuntime);
    inputGpu->dataMalloc();

    Tensor kernelGpu = 
        make_ref<TensorObj>(KERNEL_SHAPE, dtype, cudaRuntime);
    kernelGpu->dataMalloc();

    // Do memcpy host to device
    time_memcpy_htod += timeit(
        [&]() {
            inputGpu = inputCpu->clone(cudaRuntime);
            kernelGpu = kernelCpu->clone(cudaRuntime);
        },
        [&]() { cudaRuntime->sync(); }, 
        warmupRounds, timingRounds
    );

    auto []

    // get inputs
        cudnnTensorDescriptor_t inDesc;
        checkCudnnError(cudnnCreateTensorDescriptor(&inDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(
            inDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, channels, h, w));

        // get kernels
        cudnnFilterDescriptor_t knDesc;
        checkCudnnError(cudnnCreateFilterDescriptor(&knDesc));
        checkCudnnError(cudnnSetFilter4dDescriptor(knDesc, CUDNN_DATA_FLOAT,
                                                   CUDNN_TENSOR_NCHW, f,
                                                   channelsPerGrp, r, s));
        // get bias
        cudnnTensorDescriptor_t biasDesc;
        checkCudnnError(cudnnCreateTensorDescriptor(&biasDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(
            biasDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, f, 1, 1));

        // get convlution descriptor
        cudnnConvolutionDescriptor_t convDesc;
        checkCudnnError(cudnnCreateConvolutionDescriptor(&convDesc));
        // TODO: CUDNN_CONVOLUTION is a tunable argument
        checkCudnnError(cudnnSetConvolution2dDescriptor(
            convDesc, ph, pw, sh, sw, dh, dw, MODES[record->mode],
            CUDNN_DATA_FLOAT));
        if (g > 1) {
            checkCudnnError(cudnnSetConvolutionGroupCount(convDesc, g));
        }

    





    return 0;
}