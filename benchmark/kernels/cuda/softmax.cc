#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/softmax.h"
#include "benchmark.h"
#include <iostream>
#include <cmath>
#include <chrono>
#include <sys/time.h>

using namespace infini;

#define M 1048576

int main() {

    // Benchmark Settings
    int warmupRounds = 200;
    int timingRounds = 200;
    Shape INPUT_SHAPE = {16, 3, 128, 128};
    DataType dtype = DataType::Float32;

    // Get data size
    size_t size = 1;
    for (auto dim: INPUT_SHAPE) {
        size *= dim;
    }
    size_t sizeInBytes = size * sizeof(dtype);

    // Init time variables
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

    // Build input data on GPU
    Tensor inputGpu = 
        make_ref<TensorObj>(INPUT_SHAPE, dtype, cudaRuntime);
    inputGpu->dataMalloc();

    // Do memcpy host to device
    time_memcpy_htod += timeit(
        [&]() {
            inputGpu = inputCpu->clone(cudaRuntime);
        },
        [&]() { cudaRuntime->sync(); }, 
        warmupRounds, timingRounds
    );

    // Build output data on CPU
    auto outputGpu = inputGpu->clone(cudaRuntime);

    // Build output data on GPU
    Tensor outputCpu =
        make_ref<TensorObj>(INPUT_SHAPE, dtype, cpuRuntime);
    outputCpu->dataMalloc();

    // Build cudnn descriptors
    cudnnTensorDescriptor_t inputDesc, outputDesc;

    // input descriptor
    checkCudnnError(cudnnCreateTensorDescriptor(&inputDesc));
    checkCudnnError(cudnnSetTensor4dDescriptor(
        inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, INPUT_SHAPE[0],
        INPUT_SHAPE[1], INPUT_SHAPE[2], INPUT_SHAPE[3]));

    // output descriptor
    checkCudnnError(cudnnCreateTensorDescriptor(&outputDesc));
    checkCudnnError(cudnnSetTensor4dDescriptor(
        outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, INPUT_SHAPE[0],
        INPUT_SHAPE[1], INPUT_SHAPE[2], INPUT_SHAPE[3]));

    // cudnn operator settings
    float alpha = 1.0, beta = 0.0;
    cudnnSoftmaxAlgorithm_t algo = CUDNN_SOFTMAX_FAST;
    cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_INSTANCE;

    // Do forward
    time_op += timeit(
        [&]() {
            cudnnSoftmaxForward(cudaRuntime->cudnnHandle(), algo, mode, 
                &alpha, inputDesc, inputGpu->getRawDataPtr<void *>(), 
                &beta, outputDesc, outputGpu->getRawDataPtr<void *>());
        },
        [&]() { cudaRuntime->sync(); },
        warmupRounds, timingRounds
    );
    
    checkCudnnError(cudnnDestroyTensorDescriptor(inputDesc));
    checkCudnnError(cudnnDestroyTensorDescriptor(outputDesc));
    
    // Do memcpy device to host
    time_memcpy_dtoh += timeit(
        [&]() {
            outputCpu = outputGpu->clone(cpuRuntime);
        },
        [&]() { cudaRuntime->sync(); },
        warmupRounds, timingRounds
    );

    // Print Results
    printf("Operator - Softmax:\n");
    printf("Input shape: (%d, %d, %d, %d)\n", 
        INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2], INPUT_SHAPE[3]);
    printf("Input size: %ld, dtype: %s, size in bytes: %ld\n", 
        size, dtype.toString().c_str(), sizeInBytes);

    printf("TFlops: %.5lf tflops\n", 5 * size / 1e9 / time_op);
    printf("Memcpy time: h2d - %.6lf ms, d2h - %.6lf ms\n",
        time_memcpy_htod, time_memcpy_dtoh);
    printf("Memcpy throughput: h2d - %.6lf MB/ms, d2h: %.6lf MB/ms\n",
        sizeInBytes / M / time_memcpy_htod, sizeInBytes / M / time_memcpy_dtoh);    
    printf("Operation: %.6lf ms\n", time_op);

    return 0;
}