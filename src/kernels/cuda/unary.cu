#include "core/common.h"
#include "core/constants.h"
#include "cuda/cuda_common.h"
#include "cuda/cuda_unary.h"
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

namespace infini {
template <typename T> void softmax_kernel(T *input, T *output, size_t num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _softmax_kernel1<T><<<1, 1>>>(input, output, num);
    _softmax_kernel2<T><<<gridsize, blocksize>>>(input, output, num);
}
template <typename T> void relu_kernel(T *input, T *output, size_t num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _relu_kernel<T><<<gridsize, blocksize>>>(input, output, num);
}
template <typename T> void sigmoid_kernel(T *input, T *output, size_t num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _sigmoid_kernel<T><<<gridsize, blocksize>>>(input, output, num);
}
template <typename T>
void hard_sigmoid_kernel(T *input, T *output, size_t num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _hard_sigmoid_kernel<T><<<gridsize, blocksize>>>(input, output, num);
}
template <typename T> void hard_swish_kernel(T *input, T *output, size_t num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _hard_swish_kernel<T><<<gridsize, blocksize>>>(input, output, num);
}
template <typename T> void tanh_kernel(T *input, T *output, size_t num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _tanh_kernel<T><<<gridsize, blocksize>>>(input, output, num);
}
template <typename T> void abs_kernel(T *input, T *output, size_t num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _abs_kernel<T><<<gridsize, blocksize>>>(input, output, num);
}
template <typename T> void sqrt_kernel(T *input, T *output, size_t num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _sqrt_kernel<<<gridsize, blocksize>>>((T *)input, (T *)output, num);
}

template <typename T> void gelu_kernel(T *input, T *output, size_t num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _gelu_kernel<T><<<gridsize, blocksize>>>(input, output, num);
}
template <typename T> void erf_kernel(T *input, T *output, size_t num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _erf_kernel<T><<<gridsize, blocksize>>>(input, output, num);
}
template <typename T> void neg_kernel(T *input, T *output, size_t num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _neg_kernel<T><<<gridsize, blocksize>>>(input, output, num);
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

}; // namespace infini
