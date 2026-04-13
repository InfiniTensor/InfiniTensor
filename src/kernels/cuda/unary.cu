#include "core/common.h"
#include "core/constants.h"
#include "cuda/cuda_common.h"
#include "cuda/cuda_unary.h"
#include <cub/cub.cuh>
#include <math.h>

using infini::E_CONSTANT;
constexpr unsigned int num_threads() { return 32 * 4; }
constexpr int thread_work_size() { return 4; }
constexpr int block_work_size() { return thread_work_size() * num_threads(); }

template <typename T>
__global__ void _softmax_kernel1(T *input, T *output, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += pow(E_CONSTANT, input[i]);
    }
    *output = sum;
}
template <typename T>
__global__ void _softmax_kernel2(T *input, T *output, size_t n) {
    float sum = *output;
    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < n; i += stride) {
        output[i] = pow(E_CONSTANT, input[i]) / sum;
    }
}
template <typename T>
__global__ void _relu_kernel(T *input, T *output, size_t n) {
    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < n; i += stride) {
        output[i] = max(input[i], float(0));
    }
}
template <typename T>
__global__ void _sigmoid_kernel(T *input, T *output, size_t n) {
    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < n; i += stride) {
        output[i] = 1 / (1 + pow(E_CONSTANT, -input[i]));
    }
}
template <typename T>
__global__ void _hard_sigmoid_kernel(T *input, T *output, size_t n) {
    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < n; i += stride) {
        output[i] = max(0.0f, min(1.0f, 0.2f * input[i] + 0.5f));
    }
}
template <typename T>
__global__ void _hard_swish_kernel(T *input, T *output, size_t n) {
    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < n; i += stride) {
        output[i] =
            input[i] * max(0.f, min(1.f, (1.f / 6.f) * input[i] + 0.5f));
    }
}
template <typename T>
__global__ void _tanh_kernel(T *input, T *output, size_t n) {
    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < n; i += stride) {
        output[i] = (pow(E_CONSTANT, input[i]) - pow(E_CONSTANT, -input[i])) /
                    (pow(E_CONSTANT, input[i]) + pow(E_CONSTANT, -input[i]));
    }
}
template <typename T>
__global__ void _abs_kernel(T *input, T *output, size_t n) {
    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < n; i += stride) {
        output[i] = input[i] < 0 ? -input[i] : input[i];
    }
}

__global__ void _sqrt_kernel(float *input, float *output, size_t n) {
    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < n; i += stride) {
        output[i] = sqrt(input[i]);
    }
}

__global__ void _sqrt_kernel(half *input, half *output, size_t n) {
    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < n; i += stride) {
        output[i] = hsqrt(input[i]);
    }
}

__global__ void _elu_kernel(const float *input, float *output, size_t size,
                            float alpha) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        float x = input[index];
        output[index] = (x >= 0) ? x : alpha * (expf(x) - 1);
    }
}

template <typename T>
__global__ void _gelu_kernel(T *input, T *output, size_t n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        float x = input[i];
        output[i] = 0.5 * x * (1 + erf(x / sqrt(2.0f)));
    }
}

template <typename T>
__global__ void _silu_kernel(T *input, T *output, size_t n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        float x = input[i];
        output[i] = x / (1.0 + expf(-x));
    }
}

template <typename T>
__global__ void _erf_kernel(T *input, T *output, size_t n) {
    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        output[i] = erf(input[i]);
    }
}

template <typename T>
__global__ void _neg_kernel(T *input, T *output, size_t n) {
    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < n; i += stride) {
        output[i] = -input[i];
    }
}

template <typename INPUT, typename OUTPUT>
__global__ void _cast_kernel(INPUT *input, OUTPUT *output, size_t n) {

    size_t index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        cub::CastOp<OUTPUT> _CastOp;
        output[index] = _CastOp(input[index]);
    }
}

template <typename T>
__global__ void _leaky_relu_kernel(T *input, T *output, size_t n,
                                   float alphaValue) {
    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < n; i += stride) {
        output[i] = (input[i] > 0) ? input[i] : alphaValue * input[i];
    }
}

namespace infini {
template <typename T> void softmax_kernel(T *input, T *output, size_t num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _softmax_kernel1<T>
        <<<1, 1, 0, CUDAStream::getCurrentStream()>>>(input, output, num);
    _softmax_kernel2<T>
        <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(
            input, output, num);
}
template <typename T> void relu_kernel(T *input, T *output, size_t num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _relu_kernel<T><<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(
        input, output, num);
}
template <typename T> void sigmoid_kernel(T *input, T *output, size_t num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _sigmoid_kernel<T>
        <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(
            input, output, num);
}
template <typename T>
void hard_sigmoid_kernel(T *input, T *output, size_t num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _hard_sigmoid_kernel<T>
        <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(
            input, output, num);
}
template <typename T> void hard_swish_kernel(T *input, T *output, size_t num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _hard_swish_kernel<T>
        <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(
            input, output, num);
}
template <typename T> void tanh_kernel(T *input, T *output, size_t num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _tanh_kernel<T><<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(
        input, output, num);
}
template <typename T> void abs_kernel(T *input, T *output, size_t num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _abs_kernel<T><<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(
        input, output, num);
}
template <typename T> void sqrt_kernel(T *input, T *output, size_t num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _sqrt_kernel<<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(
        (T *)input, (T *)output, num);
}

template <typename T> void gelu_kernel(T *input, T *output, size_t num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _gelu_kernel<T><<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(
        input, output, num);
}

template <typename T> void silu_kernel(T *input, T *output, size_t num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _silu_kernel<T><<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(
        input, output, num);
}

template <typename T> void erf_kernel(T *input, T *output, size_t num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _erf_kernel<T><<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(
        input, output, num);
}
template <typename T> void neg_kernel(T *input, T *output, size_t num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _neg_kernel<T><<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(
        input, output, num);
}

void unary_kernel(const Operator &_op) {
    auto op = as<UnaryObj>(_op);
    void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
    void *const outputData = (op->getOutput()->getRawDataPtr<void *>());

    size_t num = op->getOutput()->size();
    if (op->getOpType() == OpType::Softmax) {
        if (_op->getDType() == DataType::Float32) {
            softmax_kernel<float>((float *)inputData, (float *)outputData, num);
        } else {
            IT_TODO_HALT();
        }
    } else if (op->getOpType() == OpType::Relu) {
        if (_op->getDType() == DataType::Float32) {
            relu_kernel<float>((float *)inputData, (float *)outputData, num);
        } else {
            IT_TODO_HALT();
        }
    } else if (op->getOpType() == OpType::Sigmoid) {
        if (_op->getDType() == DataType::Float32) {
            sigmoid_kernel<float>((float *)inputData, (float *)outputData, num);
        } else {
            IT_TODO_HALT();
        }
    } else if (op->getOpType() == OpType::HardSigmoid) {
        if (_op->getDType() == DataType::Float32) {
            hard_sigmoid_kernel<float>((float *)inputData, (float *)outputData,
                                       num);
        } else {
            IT_TODO_HALT();
        }
    } else if (op->getOpType() == OpType::HardSwish) {
        if (_op->getDType() == DataType::Float32) {
            hard_swish_kernel<float>((float *)inputData, (float *)outputData,
                                     num);
        } else {
            IT_TODO_HALT();
        }
    } else if (op->getOpType() == OpType::Tanh) {
        if (_op->getDType() == DataType::Float32) {
            tanh_kernel<float>((float *)inputData, (float *)outputData, num);
        } else {
            IT_TODO_HALT();
        }
    } else if (op->getOpType() == OpType::Abs) {
        if (_op->getDType() == DataType::Float32) {
            abs_kernel<float>((float *)inputData, (float *)outputData, num);
        } else {
            IT_TODO_HALT();
        }
    } else if (op->getOpType() == OpType::Sqrt) {
        if (_op->getDType() == DataType::Float32) {
            sqrt_kernel<float>((float *)inputData, (float *)outputData, num);
        } else if (_op->getDType() == DataType::Float16) {
            sqrt_kernel<half>((half *)inputData, (half *)outputData, num);
        } else {
            IT_TODO_HALT();
        }
    } else if (op->getOpType() == OpType::Gelu) {
        if (_op->getDType() == DataType::Float32) {
            gelu_kernel<float>((float *)inputData, (float *)outputData, num);
        } else {
            IT_TODO_HALT();
        }
    } else if (op->getOpType() == OpType::Silu) {
        if (_op->getDType() == DataType::Float32) {
            silu_kernel<float>((float *)inputData, (float *)outputData, num);
        } else if (_op->getDType() == DataType::Float16) {
            silu_kernel<half>((half *)inputData, (half *)outputData, num);
        } else {
            IT_TODO_HALT();
        }
    } else if (op->getOpType() == OpType::Neg) {
        if (_op->getDType() == DataType::Float32) {
            neg_kernel<float>((float *)inputData, (float *)outputData, num);
        } else if (_op->getDType() == DataType::Float16) {
            neg_kernel<half>((half *)inputData, (half *)outputData, num);
        } else {
            IT_TODO_HALT();
        }
    }

    else if (op->getOpType() == OpType::Erf) {
        if (_op->getDType() == DataType::Float32) {
            erf_kernel<float>((float *)inputData, (float *)outputData, num);
        } else {
            IT_TODO_HALT();
        }
    } else
        IT_TODO_HALT();
}

template <typename INPUT, typename OUTPUT>
void cast_kernel(INPUT *input, OUTPUT *output, size_t num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _cast_kernel<INPUT, OUTPUT>
        <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(
            input, output, num);
}

template <typename T>
void leaky_relu_kernel(T *input, T *output, size_t num, float alphaValue) {
    int blocksize = block_work_size();
    int gridsize = (num + blocksize - 1) / blocksize;
    _leaky_relu_kernel<<<gridsize, blocksize, 0,
                         CUDAStream::getCurrentStream()>>>(input, output, num,
                                                           alphaValue);
}

void elu_kernel(const float *input, float *output, size_t size, float alpha) {
    int blocksize = 32 * 16;
    int gridsize = (size + blocksize - 1) / blocksize;
    _elu_kernel<<<gridsize, blocksize>>>(input, output, size, alpha);
}

template void cast_kernel<float, half>(float *input, half *output, size_t num);
template void cast_kernel<half, float>(half *input, float *output, size_t num);
template void cast_kernel<float, int32_t>(float *input, int32_t *output,
                                          size_t num);
template void cast_kernel<float, int8_t>(float *input, int8_t *output,
                                         size_t num);
template void cast_kernel<int8_t, float>(int8_t *input, float *output,
                                         size_t num);
template void cast_kernel<float, bool>(float *input, bool *output,
                                         size_t num);
template void cast_kernel<long int , float>(long int  *input, float *output,
                                         size_t num);  
template void cast_kernel<long int, int >(long int  *input, int  *output,
                                         size_t num);                                   
template void leaky_relu_kernel<float>(float *input, float *output, size_t num,
                                       float alpha);
}; // namespace infini
