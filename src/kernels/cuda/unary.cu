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
__global__ void _not_kernel(const bool *input, bool *output, size_t size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        output[index] = !input[index];
    }
}
static const int MAX_PARALLEL_AXIS =
    1024; // 若 axis_dim <= 1024 使用并行 block scan，否则使用单线程顺序扫描
// --------------------------------------------------
// kernel: 每个 block 处理一个 "line"（即固定除 axis 之外的其它维度）
// data layout mapping (row-major, C-order):
//   linear index for (outer_idx, a_idx, inner_idx):
//     idx = ((outer_idx * axis_dim) + a_idx) * inner + inner_idx
// outer = product(shape[0:axis-1])
// inner = product(shape[axis+1:])
// total_lines = outer * inner
// grid.x = total_lines
// blockDim.x = (axis_dim <= MAX_PARALLEL_AXIS ? min(axis_dim, 1024) : 1)
// dynamic shared memory used when axis_dim <= MAX_PARALLEL_AXIS
// --------------------------------------------------
template <typename T>
__global__ void _cumsum_kernel(const T *__restrict__ d_in,
                               T *__restrict__ d_out, int outer, int axis_dim,
                               int inner, bool exclusive, bool reverse) {
    int line = blockIdx.x; // 0 .. outer*inner-1
    int outer_idx = line / inner;
    int inner_idx = line % inner;

    // fast path: parallel (shared mem) when axis_dim small enough
    if (axis_dim <= MAX_PARALLEL_AXIS && blockDim.x > 0) {
        extern __shared__ unsigned char smem_uc[]; // dynamic shared mem
        T *s = reinterpret_cast<T *>(
            smem_uc); // s[tid] holds element or partial result
        int tid = threadIdx.x;

        // load into shared memory (use mapping that respects reverse)
        if (tid < axis_dim) {
            int a = reverse ? (axis_dim - 1 - tid) : tid;
            size_t idx =
                (size_t)((outer_idx * (size_t)axis_dim + a) * (size_t)inner +
                         inner_idx);
            s[tid] = d_in[idx];
        }
        __syncthreads();

        // Hillis-Steele inclusive scan in-place on s[0..axis_dim-1]
        for (int offset = 1; offset < axis_dim; offset <<= 1) {
            T val = 0;
            if (tid >= offset && tid < axis_dim)
                val = s[tid - offset];
            __syncthreads();
            if (tid < axis_dim)
                s[tid] = s[tid] + val;
            __syncthreads();
        }

        // write back (apply exclusive if required)
        if (tid < axis_dim) {
            T outval;
            if (exclusive) {
                if (tid == 0)
                    outval = T(0);
                else
                    outval = s[tid - 1];
            } else {
                outval = s[tid];
            }
            int a = reverse ? (axis_dim - 1 - tid) : tid;
            size_t out_idx =
                (size_t)((outer_idx * (size_t)axis_dim + a) * (size_t)inner +
                         inner_idx);
            d_out[out_idx] = outval;
        }
    } else {
        // fallback serial path: single thread per block does sequential scan
        if (threadIdx.x == 0) {
            // forward or reverse
            if (!reverse) {
                T acc = T(0);
                for (int a = 0; a < axis_dim; ++a) {
                    size_t idx = (size_t)((outer_idx * (size_t)axis_dim + a) *
                                              (size_t)inner +
                                          inner_idx);
                    T v = d_in[idx];
                    if (exclusive) {
                        d_out[idx] = acc;
                        acc = acc + v;
                    } else {
                        acc = acc + v;
                        d_out[idx] = acc;
                    }
                }
            } else {
                T acc = T(0);
                for (int a = axis_dim - 1; a >= 0; --a) {
                    size_t idx = (size_t)((outer_idx * (size_t)axis_dim + a) *
                                              (size_t)inner +
                                          inner_idx);
                    T v = d_in[idx];
                    if (exclusive) {
                        d_out[idx] = acc;
                        acc = acc + v;
                    } else {
                        acc = acc + v;
                        d_out[idx] = acc;
                    }
                }
            }
        }
    }
}
__device__ float fast_erf(float x) {
    // 高效erf近似，精度比tanh近似更高
    float a1 = 0.254829592f;
    float a2 = -0.284496736f;
    float a3 = 1.421413741f;
    float a4 = -1.453152027f;
    float a5 = 1.061405429f;
    float p = 0.3275911f;

    int sign = x < 0 ? -1 : 1;
    x = fabsf(x);

    float t = 1.0f / (1.0f + p * x);
    float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t *
                         expf(-x * x);

    return sign * y;
}
template <typename T>
__global__ void _gelu_kernel(T *input, T *output, size_t n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        float x = input[i];
        // output[i] = 0.5 * x * (1 + erf(x / sqrt(2.0f))); //
        // 这个高精度，速度慢
        output[i] = 0.5f * x * (1.0f + fast_erf(x * 0.7071067811865475f));
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
void not_kernel(bool *input, bool *output, size_t num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _not_kernel<<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(
        input, output, num);
}

template <typename Tdata>
void cumsum_kernel(const Tdata *input, Tdata *output, int outer, int inner,
                   int dimsize, bool exclusive, bool reversive) {

    int blocks = outer * inner;
    int block_threads =
        (dimsize <= MAX_PARALLEL_AXIS) ? std::min(dimsize, 1024) : 1;
    size_t shared_bytes = (dimsize <= MAX_PARALLEL_AXIS)
                              ? (size_t)block_threads * sizeof(Tdata)
                              : 0;
    dim3 grid(blocks);
    dim3 block(block_threads);
    _cumsum_kernel<Tdata>
        <<<grid, block, shared_bytes, CUDAStream::getCurrentStream()>>>(
            input, output, outer, dimsize, inner, exclusive, reversive);
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
    } else if (op->getOpType() == OpType::Not) {
        not_kernel((bool *)inputData, (bool *)outputData, num);
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
template void cast_kernel<float, bool>(float *input, bool *output, size_t num);
template void cast_kernel<int64_t, int32_t>(int64_t *input, int32_t *output,
                                            size_t num);
template void cast_kernel<int32_t, int64_t>(int32_t *input, int64_t *output,
                                            size_t num);
template void cast_kernel<int32_t, float>(int32_t *input, float *output,
                                          size_t num);
template void cast_kernel<int64_t, float>(int64_t *input, float *output,
                                          size_t num);
template void cast_kernel<uint32_t, float>(uint32_t *input, float *output,
                                           size_t num);
template void cast_kernel<uint64_t, float>(uint64_t *input, float *output,
                                           size_t num);
template void cast_kernel<float, int64_t>(float *input, int64_t *output,
                                          size_t num);
template void cast_kernel<float, uint32_t>(float *input, uint32_t *output,
                                           size_t num);
template void cast_kernel<float, uint64_t>(float *input, uint64_t *output,
                                           size_t num);
template void cast_kernel<bool, int32_t>(bool *input, int32_t *output,
                                         size_t num);
template void cumsum_kernel<float>(const float *input, float *output, int outer,
                                   int inner, int dimsize, bool exclusive,
                                   bool reversive);
template void cumsum_kernel<half>(const half *input, half *output, int outer,
                                  int inner, int dimsize, bool exclusive,
                                  bool reversive);
template void cumsum_kernel<int32_t>(const int32_t *input, int32_t *output,
                                     int outer, int inner, int dimsize,
                                     bool exclusive, bool reversive);
template void leaky_relu_kernel<float>(float *input, float *output, size_t num,
                                       float alpha);
}; // namespace infini
