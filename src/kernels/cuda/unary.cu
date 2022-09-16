#include "core/common.h"
#include "core/constants.h"
#include "cuda/cuda_common.h"
#include <math.h>

using infini::E_CONSTANT;
constexpr unsigned int num_threads() { return 32 * 4; }
constexpr int thread_work_size() { return 4; }
constexpr int block_work_size() { return thread_work_size() * num_threads(); }

__global__ void _softmax_kernel1(float *input, float *output, int n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += pow(E_CONSTANT, input[i]);
    }
    *output = sum;
}

__global__ void _softmax_kernel2(float *input, float *output, int n) {
    float sum = *output;
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        output[i] = pow(E_CONSTANT, input[i]) / sum;
    }
}

__global__ void _relu_kernel(float *input, float *output, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        output[i] = max(input[i], float(0));
    }
}

__global__ void _sigmoid_kernel(float *input, float *output, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        output[i] = 1 / (1 + pow(E_CONSTANT, -input[i]));
    }
}

__global__ void _tanh_kernel(float *input, float *output, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        output[i] = (pow(E_CONSTANT, input[i]) - pow(E_CONSTANT, -input[i])) /
                    (pow(E_CONSTANT, input[i]) + pow(E_CONSTANT, -input[i]));
    }
}

__global__ void _abs_kernel(float *input, float *output, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        output[i] = input[i] < 0 ? -input[i] : input[i];
    }
}

namespace infini {
void softmax_kernel(float *input, float *output, int num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _softmax_kernel1<<<1, 1>>>(input, output, num);
    _softmax_kernel2<<<blocksize, gridsize>>>(input, output, num);
}
void relu_kernel(float *input, float *output, int num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _relu_kernel<<<blocksize, gridsize>>>(input, output, num);
}
void sigmoid_kernel(float *input, float *output, int num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _sigmoid_kernel<<<blocksize, gridsize>>>(input, output, num);
}
void tanh_kernel(float *input, float *output, int num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _tanh_kernel<<<blocksize, gridsize>>>(input, output, num);
}
void abs_kernel(float *input, float *output, int num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _abs_kernel<<<blocksize, gridsize>>>(input, output, num);
}

}; // namespace infini
