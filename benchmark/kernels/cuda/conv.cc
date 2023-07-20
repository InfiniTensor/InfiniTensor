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

#define M 1048576

const char algo_name[8][50] = {
    "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",
    "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
    "CUDNN_CONVOLUTION_FWD_ALGO_GEMM",
    "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",
    "CUDNN_CONVOLUTION_FWD_ALGO_FFT",
    "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",
    "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",
    "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED",
};

const char mode_name[2][50] = {
    "CUDNN_CONVOLUTION",
    "CUDNN_CROSS_CORRELATION"
};

int main() {

    int warmupRounds = 50;
    int timingRounds = 100;
    DataType dtype = DataType::Float32; 

    cudnnConvolutionMode_t convMode = CUDNN_CROSS_CORRELATION;
    cudnnConvolutionFwdAlgo_t convAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    float alpha = 1.f, beta = 0.f;

    int n, c, h, w, f, r, s;
    int INPUT_BATCH_SIZE = n = 16;
    int INPUT_CHANNELS = c = 128;
    int INPUT_HEIGHT = h = 128;
    int INPUT_WIDTH = w = 128;
    Shape INPUT_SHAPE = {INPUT_BATCH_SIZE, INPUT_CHANNELS, \
                         INPUT_HEIGHT, INPUT_WIDTH};

    int OUTPUT_CHANNELS = f = 256;
    int KERNEL_HEIGHT = r = 3;
    int KERNEL_WIDTH = s = 3;
    Shape KERNEL_SHAPE = {INPUT_CHANNELS, OUTPUT_CHANNELS, \
                          KERNEL_HEIGHT, KERNEL_WIDTH};

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

    int channelsPerGrp = INPUT_CHANNELS / NUM_GROUPS;

    // get inputs
    cudnnTensorDescriptor_t inDesc;
    checkCudnnError(cudnnCreateTensorDescriptor(&inDesc));
    checkCudnnError(cudnnSetTensor4dDescriptor(
        inDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

    // get kernels
    cudnnFilterDescriptor_t knDesc;
    checkCudnnError(cudnnCreateFilterDescriptor(&knDesc));
    checkCudnnError(cudnnSetFilter4dDescriptor(knDesc, CUDNN_DATA_FLOAT,
                                                CUDNN_TENSOR_NCHW, f,
                                                channelsPerGrp, r, s));
    
    // get bias
    // cudnnTensorDescriptor_t biasDesc;
    // checkCudnnError(cudnnCreateTensorDescriptor(&biasDesc));
    // checkCudnnError(cudnnSetTensor4dDescriptor(
    //     biasDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, f, 1, 1));

    // get convlution descriptor
    cudnnConvolutionDescriptor_t convDesc;
    checkCudnnError(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCudnnError(cudnnSetConvolution2dDescriptor(
        convDesc, PAD_HEIGHT, PAD_WIDTH, VERTICAL_STRIDE, HORIZONTAL_STRIDE, 
        DILATION_HEIGHT, DILATION_WIDTH, convMode, CUDNN_DATA_FLOAT));
    if (NUM_GROUPS > 1) {
        checkCudnnError(cudnnSetConvolutionGroupCount(convDesc, NUM_GROUPS));
    }

    int outn, outc, outh, outw;
    checkCudnnError(cudnnGetConvolution2dForwardOutputDim(
        convDesc, inDesc, knDesc, &outn, &outc, &outh, &outw));

    cudnnTensorDescriptor_t outDesc;
    checkCudnnError(cudnnCreateTensorDescriptor(&outDesc));
    checkCudnnError(cudnnSetTensor4dDescriptor(outDesc, CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_FLOAT, outn, outc,
                                               outh, outw));

    Shape OUTPUT_SHAPE = {outn, outc, outh, outw};
    size_t outputSize = 1;
    for (auto dim: OUTPUT_SHAPE) {
        outputSize *= dim;
    }
    size_t outputSizeInBytes = outputSize * sizeof(dtype);

    // Build output data on CPU
    Tensor outputCpu =
        make_ref<TensorObj>(OUTPUT_SHAPE, dtype, cpuRuntime);
    outputCpu->dataMalloc();

    // Build output data on GPU
    Tensor outputGpu = 
        make_ref<TensorObj>(OUTPUT_SHAPE, dtype, cudaRuntime);
    outputGpu->dataMalloc();

    size_t workspaceSize = 0;
    checkCudnnError(cudnnGetConvolutionForwardWorkspaceSize(
        cudaRuntime->cudnnHandle(), inDesc, knDesc, convDesc,
        outDesc, convAlgo, &workspaceSize));

    CudaPtr workspace = cudaRuntime->getWorkspace(workspaceSize);

    time_op += timeit(
        [&]() {
            cudnnConvolutionForward(cudaRuntime->cudnnHandle(), &alpha,
                                    inDesc, inputGpu->getRawDataPtr<void *>(), 
                                    knDesc, kernelGpu->getRawDataPtr<void *>(),
                                    convDesc, convAlgo, workspace, 
                                    workspaceSize, &beta, 
                                    outDesc, outputGpu->getRawDataPtr<void *>());
        },
        [&]() { cudaRuntime->sync(); },
        warmupRounds, timingRounds
    );

    checkCudnnError(cudnnDestroyTensorDescriptor(outDesc));
    checkCudnnError(cudnnDestroyConvolutionDescriptor(convDesc));
    // checkCudnnError(cudnnDestroyTensorDescriptor(biasDesc));
    checkCudnnError(cudnnDestroyFilterDescriptor(knDesc));
    checkCudnnError(cudnnDestroyTensorDescriptor(inDesc));

    time_memcpy_dtoh += timeit(
        [&]() {
            outputCpu = outputGpu->clone(cpuRuntime);
        },
        [&]() { cudaRuntime->sync(); },
        warmupRounds, timingRounds
    );

    // Print Results
    printf("Operator - Convolution:\n");
    printf("Conv Algo: %s\n", algo_name[convAlgo]);
    printf("Conv Mode: %s\n", mode_name[convMode]);
    printf("Input shape: (%d, %d, %d, %d)\n", 
        INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2], INPUT_SHAPE[3]);
    printf("Kernel shape: (%d, %d, %d, %d)\n", 
        KERNEL_SHAPE[0], KERNEL_SHAPE[1], KERNEL_SHAPE[2], KERNEL_SHAPE[3]);
    printf("Output shape: (%d, %d, %d, %d)\n", 
        OUTPUT_SHAPE[0], OUTPUT_SHAPE[1], OUTPUT_SHAPE[2], OUTPUT_SHAPE[3]);     
    printf("Workspace size: %ld Bytes, dtype: %s\n", 
        workspaceSize, dtype.toString().c_str());

    printf("TFlops: %.5lf tflops\n",
        2.0 * INPUT_BATCH_SIZE * channelsPerGrp * outh * outw * \
              OUTPUT_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH / \
              VERTICAL_STRIDE / HORIZONTAL_STRIDE / 1e9 / time_op);
    printf("Memcpy time: h2d - %.6lf ms, d2h - %.6lf ms\n",
        time_memcpy_htod, time_memcpy_dtoh);
    printf("Memcpy throughput: h2d - %.6lf MB/ms, d2h: %.6lf MB/ms\n",
        (inputSizeInBytes + kernelSizeInBytes) / M / time_memcpy_htod, 
        outputSizeInBytes / M / time_memcpy_dtoh);    
    printf("Operation: %.6lf ms\n", time_op);

    return 0;
}