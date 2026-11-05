#include "cuda/cuda_common.h"
#include "cuda/cuda_element_wise.h"
#include "cuda/cuda_utility.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>
#include <limits>
#include <math.h>
#include <stdexcept>
#include <string>

constexpr unsigned int num_threads() { return 32 * 4; }
constexpr int thread_work_size() { return 4; }
constexpr int block_work_size() { return thread_work_size() * num_threads(); }

enum class BinaryOp { Add, Sub, Mul };

template <BinaryOp Op, class T>
__device__ __forceinline__ T binaryValue(T a, T b) {
    if constexpr (Op == BinaryOp::Add)
        return a + b;
    else if constexpr (Op == BinaryOp::Sub)
        return a - b;
    else
        return a * b;
}

template <BinaryOp Op>
__device__ __forceinline__ half binaryHalf(half a, half b) {
    if constexpr (Op == BinaryOp::Add)
        return __hadd(a, b);
    else if constexpr (Op == BinaryOp::Sub)
        return __hsub(a, b);
    else
        return __hmul(a, b);
}

template <BinaryOp Op>
__device__ __forceinline__ half2 binaryHalf2(half2 a, half2 b) {
    if constexpr (Op == BinaryOp::Add)
        return __hadd2(a, b);
    else if constexpr (Op == BinaryOp::Sub)
        return __hsub2(a, b);
    else
        return __hmul2(a, b);
}

template <BinaryOp Op> constexpr const char *binaryOpName() {
    if constexpr (Op == BinaryOp::Add)
        return "Add";
    else if constexpr (Op == BinaryOp::Sub)
        return "Sub";
    else
        return "Mul";
}

template <class T>
__global__ void _div_kernel(void *x, void *y, void *z, int a0, int a1, int a2,
                            int a3, int b0, int b1, int b2, int b3, int c0,
                            int c1, int c2, int c3) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int n = c0 * c1 * c2 * c3;

    for (int i = index; i < n; i += stride) {
        int c0_index = i / (c1 * c2 * c3);
        int c1_index = (i % (c1 * c2 * c3)) / (c2 * c3);
        int c2_index = ((i % (c1 * c2 * c3)) % (c2 * c3)) / c3;
        int c3_index = ((i % (c1 * c2 * c3)) % (c2 * c3)) % c3;

        int a0_index = c0_index % a0;
        int a1_index = c1_index % a1;
        int a2_index = c2_index % a2;
        int a3_index = c3_index % a3;

        int b0_index = c0_index % b0;
        int b1_index = c1_index % b1;
        int b2_index = c2_index % b2;
        int b3_index = c3_index % b3;
        ((T *)z)[i] = ((T *)x)[a0_index * a1 * a2 * a3 + a1_index * a2 * a3 +
                               a2_index * a3 + a3_index] /
                      ((T *)y)[b0_index * b1 * b2 * b3 + b1_index * b2 * b3 +
                               b2_index * b3 + b3_index];
    }
}

template <class T>
__global__ void _add_kernel(void *x, void *y, void *z, int a0, int a1, int a2,
                            int a3, int b0, int b1, int b2, int b3, int c0,
                            int c1, int c2, int c3) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int n = c0 * c1 * c2 * c3;

    for (int i = index; i < n; i += stride) {
        int c0_index = i / (c1 * c2 * c3);
        int c1_index = (i % (c1 * c2 * c3)) / (c2 * c3);
        int c2_index = ((i % (c1 * c2 * c3)) % (c2 * c3)) / c3;
        int c3_index = ((i % (c1 * c2 * c3)) % (c2 * c3)) % c3;

        int a0_index = c0_index % a0;
        int a1_index = c1_index % a1;
        int a2_index = c2_index % a2;
        int a3_index = c3_index % a3;

        int b0_index = c0_index % b0;
        int b1_index = c1_index % b1;
        int b2_index = c2_index % b2;
        int b3_index = c3_index % b3;
        ((T *)z)[i] = ((T *)x)[a0_index * a1 * a2 * a3 + a1_index * a2 * a3 +
                               a2_index * a3 + a3_index] +
                      ((T *)y)[b0_index * b1 * b2 * b3 + b1_index * b2 * b3 +
                               b2_index * b3 + b3_index];
    }
}

template <class T>
__global__ void _sub_kernel(void *x, void *y, void *z, int a0, int a1, int a2,
                            int a3, int b0, int b1, int b2, int b3, int c0,
                            int c1, int c2, int c3) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int n = c0 * c1 * c2 * c3;

    for (int i = index; i < n; i += stride) {
        int c0_index = i / (c1 * c2 * c3);
        int c1_index = (i % (c1 * c2 * c3)) / (c2 * c3);
        int c2_index = ((i % (c1 * c2 * c3)) % (c2 * c3)) / c3;
        int c3_index = ((i % (c1 * c2 * c3)) % (c2 * c3)) % c3;

        int a0_index = c0_index % a0;
        int a1_index = c1_index % a1;
        int a2_index = c2_index % a2;
        int a3_index = c3_index % a3;

        int b0_index = c0_index % b0;
        int b1_index = c1_index % b1;
        int b2_index = c2_index % b2;
        int b3_index = c3_index % b3;
        ((T *)z)[i] = ((T *)x)[a0_index * a1 * a2 * a3 + a1_index * a2 * a3 +
                               a2_index * a3 + a3_index] -
                      ((T *)y)[b0_index * b1 * b2 * b3 + b1_index * b2 * b3 +
                               b2_index * b3 + b3_index];
    }
}

template <class T, BinaryOp Op = BinaryOp::Mul>
__global__ void _binary_broadcast_kernel(void *x, void *y, void *z, int a0,
                                         int a1, int a2, int a3, int b0, int b1,
                                         int b2, int b3, int c0, int c1, int c2,
                                         int c3) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int n = c0 * c1 * c2 * c3;

    for (int i = index; i < n; i += stride) {
        int c0_index = i / (c1 * c2 * c3);
        int c1_index = (i % (c1 * c2 * c3)) / (c2 * c3);
        int c2_index = ((i % (c1 * c2 * c3)) % (c2 * c3)) / c3;
        int c3_index = ((i % (c1 * c2 * c3)) % (c2 * c3)) % c3;

        int a0_index = c0_index % a0;
        int a1_index = c1_index % a1;
        int a2_index = c2_index % a2;
        int a3_index = c3_index % a3;

        int b0_index = c0_index % b0;
        int b1_index = c1_index % b1;
        int b2_index = c2_index % b2;
        int b3_index = c3_index % b3;
        ((T *)z)[i] = binaryValue<Op>(
            ((T *)x)[a0_index * a1 * a2 * a3 + a1_index * a2 * a3 +
                     a2_index * a3 + a3_index],
            ((T *)y)[b0_index * b1 * b2 * b3 + b1_index * b2 * b3 +
                     b2_index * b3 + b3_index]);
    }
}

// 形状相同的 FP32 输入无需计算广播地址
template <BinaryOp Op>
__global__ void _binary_float_no_broadcast_kernel(const float *a,
                                                  const float *b, float *c,
                                                  int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = binaryValue<Op>(a[i], b[i]);
}


template <BinaryOp Op>
__global__ void _binary_half_no_broadcast_pair_kernel(const half *a,
                                                      const half *b, half *c,
                                                      int n) {
    const int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        c[i] = binaryHalf<Op>(a[i], b[i]);
        if (i + 1 < n)
            c[i + 1] = binaryHalf<Op>(a[i + 1], b[i + 1]);
    }
}

template <BinaryOp Op>
__global__ void _binary_half_no_broadcast_half2_kernel(const half *a,
                                                       const half *b, half *c,
                                                       int n) {
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair < n / 2)
        reinterpret_cast<half2 *>(c)[pair] =
            binaryHalf2<Op>(reinterpret_cast<const half2 *>(a)[pair],
                            reinterpret_cast<const half2 *>(b)[pair]);
    else if ((n & 1) && pair == n / 2)
        c[n - 1] = binaryHalf<Op>(a[n - 1], b[n - 1]);
}

struct BinaryBroadcastShape {
    int a[4], b[4], c[4];
};


__device__ __forceinline__ int binaryBroadcastOffset(int i, const int *input,
                                                     const int *output) {
    const int c0 = i / (output[1] * output[2] * output[3]);
    const int c1 =
        (i % (output[1] * output[2] * output[3])) / (output[2] * output[3]);
    const int c2 =
        ((i % (output[1] * output[2] * output[3])) % (output[2] * output[3])) /
        output[3];
    const int c3 =
        ((i % (output[1] * output[2] * output[3])) % (output[2] * output[3])) %
        output[3];
    return (c0 % input[0]) * input[1] * input[2] * input[3] +
           (c1 % input[1]) * input[2] * input[3] + (c2 % input[2]) * input[3] +
           c3 % input[3];
}

template <BinaryOp Op>
__global__ void _binary_half_broadcast_pair_kernel(const half *a, const half *b,
                                                   half *c, int n,
                                                   BinaryBroadcastShape shape) {
    const int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n)
        return;
    const int a0 = binaryBroadcastOffset(i, shape.a, shape.c);
    const int b0 = binaryBroadcastOffset(i, shape.b, shape.c);
    if (i + 1 >= n) {
        c[i] = binaryHalf<Op>(a[a0], b[b0]);
        return;
    }
    const int a1 = binaryBroadcastOffset(i + 1, shape.a, shape.c);
    const int b1 = binaryBroadcastOffset(i + 1, shape.b, shape.c);

    const half result0 = binaryHalf<Op>(a[a0], b[b0]);
    const half result1 = binaryHalf<Op>(a[a1], b[b1]);
    c[i] = result0;
    c[i + 1] = result1;
}

template <class T>
__global__ void _min_kernel(void *x, void *y, void *z, int a0, int a1, int a2,
                            int a3, int b0, int b1, int b2, int b3, int c0,
                            int c1, int c2, int c3) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int n = c0 * c1 * c2 * c3;

    for (int i = index; i < n; i += stride) {
        int c0_index = i / (c1 * c2 * c3);
        int c1_index = (i % (c1 * c2 * c3)) / (c2 * c3);
        int c2_index = ((i % (c1 * c2 * c3)) % (c2 * c3)) / c3;
        int c3_index = ((i % (c1 * c2 * c3)) % (c2 * c3)) % c3;

        int a0_index = c0_index % a0;
        int a1_index = c1_index % a1;
        int a2_index = c2_index % a2;
        int a3_index = c3_index % a3;

        int b0_index = c0_index % b0;
        int b1_index = c1_index % b1;
        int b2_index = c2_index % b2;
        int b3_index = c3_index % b3;
        T a_val = ((T *)x)[a0_index * a1 * a2 * a3 + a1_index * a2 * a3 +
                           a2_index * a3 + a3_index];
        T b_val = ((T *)y)[b0_index * b1 * b2 * b3 + b1_index * b2 * b3 +
                           b2_index * b3 + b3_index];
        ((T *)z)[i] = a_val < b_val ? a_val : b_val;
    }
}

template <class T>
__global__ void _max_kernel(void *x, void *y, void *z, int a0, int a1, int a2,
                            int a3, int b0, int b1, int b2, int b3, int c0,
                            int c1, int c2, int c3) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int n = c0 * c1 * c2 * c3;

    for (int i = index; i < n; i += stride) {
        int c0_index = i / (c1 * c2 * c3);
        int c1_index = (i % (c1 * c2 * c3)) / (c2 * c3);
        int c2_index = ((i % (c1 * c2 * c3)) % (c2 * c3)) / c3;
        int c3_index = ((i % (c1 * c2 * c3)) % (c2 * c3)) % c3;

        int a0_index = c0_index % a0;
        int a1_index = c1_index % a1;
        int a2_index = c2_index % a2;
        int a3_index = c3_index % a3;

        int b0_index = c0_index % b0;
        int b1_index = c1_index % b1;
        int b2_index = c2_index % b2;
        int b3_index = c3_index % b3;
        T a_val = ((T *)x)[a0_index * a1 * a2 * a3 + a1_index * a2 * a3 +
                           a2_index * a3 + a3_index];
        T b_val = ((T *)y)[b0_index * b1 * b2 * b3 + b1_index * b2 * b3 +
                           b2_index * b3 + b3_index];
        ((T *)z)[i] = a_val > b_val ? a_val : b_val;
    }
}

template <class T>
__global__ void _pow_kernel(void *x, void *y, void *z, int a0, int a1, int a2,
                            int a3, int b0, int b1, int b2, int b3, int c0,
                            int c1, int c2, int c3) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int n = c0 * c1 * c2 * c3;

    for (int i = index; i < n; i += stride) {
        int c0_index = i / (c1 * c2 * c3);
        int c1_index = (i % (c1 * c2 * c3)) / (c2 * c3);
        int c2_index = ((i % (c1 * c2 * c3)) % (c2 * c3)) / c3;
        int c3_index = ((i % (c1 * c2 * c3)) % (c2 * c3)) % c3;

        int a0_index = c0_index % a0;
        int a1_index = c1_index % a1;
        int a2_index = c2_index % a2;
        int a3_index = c3_index % a3;

        int b0_index = c0_index % b0;
        int b1_index = c1_index % b1;
        int b2_index = c2_index % b2;
        int b3_index = c3_index % b3;
        float base = static_cast<float>(
            ((T *)x)[a0_index * a1 * a2 * a3 + a1_index * a2 * a3 +
                     a2_index * a3 + a3_index]);
        float exponent = static_cast<float>(
            ((T *)y)[b0_index * b1 * b2 * b3 + b1_index * b2 * b3 +
                     b2_index * b3 + b3_index]);
        ((T *)z)[i] = powf(base, exponent);
    }
}

template <class T>
__global__ void _less_kernel(void *x, void *y, void *z, int a0, int a1, int a2,
                             int a3, int b0, int b1, int b2, int b3, int c0,
                             int c1, int c2, int c3) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int n = c0 * c1 * c2 * c3;

    for (int i = index; i < n; i += stride) {
        int c0_index = i / (c1 * c2 * c3);
        int c1_index = (i % (c1 * c2 * c3)) / (c2 * c3);
        int c2_index = ((i % (c1 * c2 * c3)) % (c2 * c3)) / c3;
        int c3_index = ((i % (c1 * c2 * c3)) % (c2 * c3)) % c3;

        int a0_index = c0_index % a0;
        int a1_index = c1_index % a1;
        int a2_index = c2_index % a2;
        int a3_index = c3_index % a3;

        int b0_index = c0_index % b0;
        int b1_index = c1_index % b1;
        int b2_index = c2_index % b2;
        int b3_index = c3_index % b3;
        ((bool *)z)[i] =
            ((T *)x)[a0_index * a1 * a2 * a3 + a1_index * a2 * a3 +
                     a2_index * a3 + a3_index] <
                    ((T *)y)[b0_index * b1 * b2 * b3 + b1_index * b2 * b3 +
                             b2_index * b3 + b3_index]
                ? true
                : false;
    }
}

#define CASE(OP, T)                                                            \
    _##OP##_kernel<DT_CUDA<T>::t>                                              \
        <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(          \
            a, b, c, a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3);

#define SWITCH_DTYPE(OP, DTYPE)                                                \
    switch (DTYPE) {                                                           \
    case 1:                                                                    \
        CASE(OP, 1)                                                            \
        break;                                                                 \
    case 2:                                                                    \
        CASE(OP, 2)                                                            \
        break;                                                                 \
    case 3:                                                                    \
        CASE(OP, 3)                                                            \
        break;                                                                 \
    case 4:                                                                    \
        CASE(OP, 4)                                                            \
        break;                                                                 \
    case 5:                                                                    \
        CASE(OP, 5)                                                            \
        break;                                                                 \
    case 6:                                                                    \
        CASE(OP, 6)                                                            \
        break;                                                                 \
    case 7:                                                                    \
        CASE(OP, 7)                                                            \
        break;                                                                 \
    case 10:                                                                   \
        CASE(OP, 10)                                                           \
        break;                                                                 \
    case 11:                                                                   \
        CASE(OP, 11)                                                           \
        break;                                                                 \
    case 12:                                                                   \
        CASE(OP, 12)                                                           \
        break;                                                                 \
    case 13:                                                                   \
        CASE(OP, 13)                                                           \
        break;                                                                 \
    case 16:                                                                   \
        CASE(OP, 16)                                                           \
        break;                                                                 \
    default:                                                                   \
        IT_TODO_HALT();                                                        \
    }

template <class T>
__global__ void _div_const_kernel(void const *__restrict__ x,
                                  void const *__restrict__ y,
                                  void *__restrict__ z, const size_t n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        ((T *)z)[tid] = ((T *)x)[tid] / *((T *)y);
    }
}

template <class T>
__global__ void _pow_const_kernel(void const *__restrict__ x,
                                  void const *__restrict__ y,
                                  void *__restrict__ z, const size_t n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        ((T *)z)[tid] = pow(((T *)x)[tid], *((T *)y));
    }
}
template <>
__global__ void _pow_const_kernel<half>(void const *__restrict__ x,
                                        void const *__restrict__ y,
                                        void *__restrict__ z, const size_t n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
#ifdef USE_METAX
        ((half *)z)[tid] =
            (half)powf(((float)((half *)x)[tid]), (float)*((half *)y));
#else
        ((half *)z)[tid] = pow(((float)((half *)x)[tid]), (float)*((half *)y));
#endif
    }
}

#define CASE_CONST(OP, T)                                                      \
    _##OP##_const_kernel<DT_CUDA<T>::t>                                        \
        <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(a, b, c,  \
                                                                     n);

#define SWITCH_DTYPE_CONST(OP, DTYPE)                                          \
    switch (DTYPE) {                                                           \
    case 1:                                                                    \
        CASE_CONST(OP, 1)                                                      \
        break;                                                                 \
    case 2:                                                                    \
        CASE_CONST(OP, 2)                                                      \
        break;                                                                 \
    case 3:                                                                    \
        CASE_CONST(OP, 3)                                                      \
        break;                                                                 \
    case 4:                                                                    \
        CASE_CONST(OP, 4)                                                      \
        break;                                                                 \
    case 5:                                                                    \
        CASE_CONST(OP, 5)                                                      \
        break;                                                                 \
    case 6:                                                                    \
        CASE_CONST(OP, 6)                                                      \
        break;                                                                 \
    case 7:                                                                    \
        CASE_CONST(OP, 7)                                                      \
        break;                                                                 \
    case 10:                                                                   \
        CASE_CONST(OP, 10)                                                     \
        break;                                                                 \
    case 11:                                                                   \
        CASE_CONST(OP, 11)                                                     \
        break;                                                                 \
    case 12:                                                                   \
        CASE_CONST(OP, 12)                                                     \
        break;                                                                 \
    case 13:                                                                   \
        CASE_CONST(OP, 13)                                                     \
        break;                                                                 \
    default:                                                                   \
        IT_TODO_HALT();                                                        \
    }

namespace infini {
namespace {
template <typename Mode> struct MulModeEntry {
    Mode mode;
    const char *name;
    const char *legacy;
};

constexpr MulModeEntry<Fp16MulMode> fp16MulModes[] = {
    {Fp16MulMode::Base, "base", nullptr},
    {Fp16MulMode::BroadcastPair, "broadcast_pair", "generic_pair"},
    {Fp16MulMode::NoBroadcastUnaligned, "no_broadcast_unaligned", "unaligned"},
    {Fp16MulMode::NoBroadcastHalf2, "no_broadcast_half2", "half2"},
    {Fp16MulMode::Auto, "auto", nullptr}};
constexpr MulModeEntry<Fp32MulMode> fp32MulModes[] = {
    {Fp32MulMode::Broadcast, "broadcast", "base"},
    {Fp32MulMode::NoBroadcast, "no_broadcast", "linear"},
    {Fp32MulMode::Auto, "auto", nullptr}};

template <typename Mode, size_t N>
const char *lookupMulModeName(Mode mode, const MulModeEntry<Mode> (&modes)[N]) {
    for (const auto &entry : modes)
        if (entry.mode == mode)
            return entry.name;
    return "invalid";
}

template <typename Mode, size_t N>
Mode parseMulMode(const char *value, const MulModeEntry<Mode> (&modes)[N],
                  const char *error) {
    if (!value || !*value)
        return Mode::Auto;
    for (const auto &entry : modes)
        if (std::strcmp(value, entry.name) == 0 ||
            (entry.legacy && std::strcmp(value, entry.legacy) == 0))
            return entry.mode;
    throw std::invalid_argument(error);
}

int validateBinaryBroadcastAndSize(const std::array<int, 4> &a,
                                   const std::array<int, 4> &b,
                                   const std::array<int, 4> &c, int limit,
                                   const char *precision,
                                   const char *operation) {

    int n = 1;
    for (size_t axis = 0; axis < c.size(); ++axis) {
        if (a[axis] <= 0 || b[axis] <= 0 || c[axis] <= 0 ||
            (a[axis] != 1 && a[axis] != c[axis]) ||
            (b[axis] != 1 && b[axis] != c[axis]) ||
            c[axis] != (a[axis] > b[axis] ? a[axis] : b[axis]))
            throw std::invalid_argument(std::string(precision) + " " +
                                        operation + ": invalid broadcast");
        if (n > limit / c[axis])
            throw std::invalid_argument(std::string(precision) + " " +
                                        operation + ": tensor too large");
        n *= c[axis];
    }
    return n;
}
} // namespace

const char *mulModeName(Fp16MulMode mode) {
    return lookupMulModeName(mode, fp16MulModes);
}
const char *mulModeName(Fp32MulMode mode) {
    return lookupMulModeName(mode, fp32MulModes);
}
Fp16MulMode parseFp16MulMode(const char *value) {
    return parseMulMode(value, fp16MulModes,
                        "INFINITENSOR_FP16_MUL_MODE: expected "
                        "auto|base|broadcast_pair|"
                        "no_broadcast_unaligned|no_broadcast_half2 "
                        "(aliases: generic_pair|unaligned|half2)");
}
Fp32MulMode parseFp32MulMode(const char *value) {
    return parseMulMode(value, fp32MulModes,
                        "INFINITENSOR_FP32_MUL_MODE: expected "
                        "auto|broadcast|no_broadcast");
}

namespace {
template <BinaryOp Op>
Fp16MulMode dispatchBinary(Fp16MulMode requested, void *a, void *b, void *c,
                           const std::array<int, 4> &aDims,
                           const std::array<int, 4> &bDims,
                           const std::array<int, 4> &cDims) {
    if (requested < Fp16MulMode::Base || requested > Fp16MulMode::Auto)
        throw std::invalid_argument(std::string("FP16 ") + binaryOpName<Op>() +
                                    ": unknown mode");
    const int n = validateBinaryBroadcastAndSize(
        aDims, bDims, cDims,
        std::numeric_limits<int>::max() - block_work_size(), "FP16",
        binaryOpName<Op>());
    if (!a || !b || !c)
        throw std::invalid_argument(std::string("FP16 ") + binaryOpName<Op>() +
                                    ": null buffer");
    auto actual = requested;
    const bool broadcast = aDims != cDims || bDims != cDims;
    const bool aligned = (reinterpret_cast<std::uintptr_t>(a) |
                          reinterpret_cast<std::uintptr_t>(b) |
                          reinterpret_cast<std::uintptr_t>(c)) %
                             alignof(half2) ==
                         0;
    if (actual == Fp16MulMode::Auto)
        actual = broadcast ? Fp16MulMode::BroadcastPair
                 : aligned ? Fp16MulMode::NoBroadcastHalf2
                           : Fp16MulMode::NoBroadcastUnaligned;
    if ((requested == Fp16MulMode::NoBroadcastUnaligned ||
         requested == Fp16MulMode::NoBroadcastHalf2) &&
        broadcast)
        actual = Fp16MulMode::BroadcastPair;
    if (actual == Fp16MulMode::NoBroadcastHalf2 && !aligned)
        actual = Fp16MulMode::NoBroadcastUnaligned;
    const int block = block_work_size();
    const int grid = n / block + (n % block != 0);
    const auto stream = CUDAStream::getCurrentStream();
    const auto *inputA = static_cast<const half *>(a);
    const auto *inputB = static_cast<const half *>(b);
    auto *output = static_cast<half *>(c);
    if (actual == Fp16MulMode::Base) {
        _binary_broadcast_kernel<half, Op><<<grid, block, 0, stream>>>(
            a, b, c, aDims[0], aDims[1], aDims[2], aDims[3], bDims[0], bDims[1],
            bDims[2], bDims[3], cDims[0], cDims[1], cDims[2], cDims[3]);
    } else {
        const int pairs = n / 2 + (n & 1);
        const int pairGrid = pairs / block + (pairs % block != 0);
        if (actual == Fp16MulMode::BroadcastPair) {
            const BinaryBroadcastShape shape{
                {aDims[0], aDims[1], aDims[2], aDims[3]},
                {bDims[0], bDims[1], bDims[2], bDims[3]},
                {cDims[0], cDims[1], cDims[2], cDims[3]}};
            _binary_half_broadcast_pair_kernel<Op>
                <<<pairGrid, block, 0, stream>>>(inputA, inputB, output, n,
                                                 shape);
        } else if (actual == Fp16MulMode::NoBroadcastUnaligned) {
            _binary_half_no_broadcast_pair_kernel<Op>
                <<<pairGrid, block, 0, stream>>>(inputA, inputB, output, n);
        } else {
            _binary_half_no_broadcast_half2_kernel<Op>
                <<<pairGrid, block, 0, stream>>>(inputA, inputB, output, n);
        }
    }
    return actual;
}

template <BinaryOp Op>
Fp32MulMode dispatchBinary(Fp32MulMode requested, void *a, void *b, void *c,
                           const std::array<int, 4> &aDims,
                           const std::array<int, 4> &bDims,
                           const std::array<int, 4> &cDims) {
    if (requested < Fp32MulMode::Broadcast || requested > Fp32MulMode::Auto)
        throw std::invalid_argument(std::string("FP32 ") + binaryOpName<Op>() +
                                    ": unknown mode");
    const int n = validateBinaryBroadcastAndSize(
        aDims, bDims, cDims,
        (std::numeric_limits<int>::max() - block_work_size()) / 2, "FP32",
        binaryOpName<Op>());
    if (!a || !b || !c)
        throw std::invalid_argument(std::string("FP32 ") + binaryOpName<Op>() +
                                    ": null buffer");
    const bool broadcast = aDims != cDims || bDims != cDims;
    const Fp32MulMode actual = requested != Fp32MulMode::Broadcast && !broadcast
                                   ? Fp32MulMode::NoBroadcast
                                   : Fp32MulMode::Broadcast;
    const int block = block_work_size();
    const int grid = n / block + (n % block != 0);
    const auto stream = CUDAStream::getCurrentStream();
    if (actual == Fp32MulMode::NoBroadcast) {
        _binary_float_no_broadcast_kernel<Op><<<grid, block, 0, stream>>>(
            static_cast<const float *>(a), static_cast<const float *>(b),
            static_cast<float *>(c), n);
    } else {
        _binary_broadcast_kernel<float, Op><<<grid, block, 0, stream>>>(
            a, b, c, aDims[0], aDims[1], aDims[2], aDims[3], bDims[0], bDims[1],
            bDims[2], bDims[3], cDims[0], cDims[1], cDims[2], cDims[3]);
    }
    return actual;
}

template <BinaryOp Op>
bool launchBinaryAuto(int dType, void *a, void *b, void *c,
                      const std::array<int, 4> &aDims,
                      const std::array<int, 4> &bDims,
                      const std::array<int, 4> &cDims) {
    if (dType == 10)
        dispatchBinary<Op>(Fp16MulMode::Auto, a, b, c, aDims, bDims, cDims);
    else if (dType == 1)
        dispatchBinary<Op>(Fp32MulMode::Auto, a, b, c, aDims, bDims, cDims);
    else
        return false;
    return true;
}
} // namespace

Fp16MulMode dispatchMul(Fp16MulMode requested, void *a, void *b, void *c,
                        const std::array<int, 4> &aDims,
                        const std::array<int, 4> &bDims,
                        const std::array<int, 4> &cDims) {
    return dispatchBinary<BinaryOp::Mul>(requested, a, b, c, aDims, bDims,
                                         cDims);
}

Fp32MulMode dispatchMul(Fp32MulMode requested, void *a, void *b, void *c,
                        const std::array<int, 4> &aDims,
                        const std::array<int, 4> &bDims,
                        const std::array<int, 4> &cDims) {
    return dispatchBinary<BinaryOp::Mul>(requested, a, b, c, aDims, bDims,
                                         cDims);
}

void div_const_kernel(int dType, void *a, void *b, void *c, size_t n) {
    size_t blocksize = block_work_size();
    size_t gridsize = (n + block_work_size() - 1) / block_work_size();
    SWITCH_DTYPE_CONST(div, dType);
}

void pow_const_kernel(int dType, void *a, void *b, void *c, size_t n) {
    size_t blocksize = block_work_size();
    size_t gridsize = (n + block_work_size() - 1) / block_work_size();
    SWITCH_DTYPE_CONST(pow, dType);
}

void div_kernel(int dType, void *a, void *b, void *c, int a0, int a1, int a2,
                int a3, int b0, int b1, int b2, int b3, int c0, int c1, int c2,
                int c3) {

    int blocksize = block_work_size();
    int num = c0 * c1 * c2 * c3;
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    SWITCH_DTYPE(div, dType)
}
void add_kernel(int dType, void *a, void *b, void *c, int a0, int a1, int a2,
                int a3, int b0, int b1, int b2, int b3, int c0, int c1, int c2,
                int c3) {
    if (launchBinaryAuto<BinaryOp::Add>(dType, a, b, c, {a0, a1, a2, a3},
                                        {b0, b1, b2, b3}, {c0, c1, c2, c3}))
        return;
    int blocksize = block_work_size();
    int num = c0 * c1 * c2 * c3;
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    SWITCH_DTYPE(add, dType)
}
void sub_kernel(int dType, void *a, void *b, void *c, int a0, int a1, int a2,
                int a3, int b0, int b1, int b2, int b3, int c0, int c1, int c2,
                int c3) {
    if (launchBinaryAuto<BinaryOp::Sub>(dType, a, b, c, {a0, a1, a2, a3},
                                        {b0, b1, b2, b3}, {c0, c1, c2, c3}))
        return;
    int blocksize = block_work_size();
    int num = c0 * c1 * c2 * c3;
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    SWITCH_DTYPE(sub, dType)
}
void mul_kernel(int dType, void *a, void *b, void *c, int a0, int a1, int a2,
                int a3, int b0, int b1, int b2, int b3, int c0, int c1, int c2,
                int c3) {

    if (dType == 10) {
        static const auto requested =
            parseFp16MulMode(std::getenv("INFINITENSOR_FP16_MUL_MODE"));
        if (requested != Fp16MulMode::Base) {
            dispatchMul(requested, a, b, c, {a0, a1, a2, a3}, {b0, b1, b2, b3},
                        {c0, c1, c2, c3});
            return;
        }
    }
    if (dType == 1) {
        static const auto requested =
            parseFp32MulMode(std::getenv("INFINITENSOR_FP32_MUL_MODE"));
        if (requested != Fp32MulMode::Broadcast) {
            dispatchMul(requested, a, b, c, {a0, a1, a2, a3}, {b0, b1, b2, b3},
                        {c0, c1, c2, c3});
            return;
        }
    }
    int blocksize = block_work_size();
    int num = c0 * c1 * c2 * c3;
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    SWITCH_DTYPE(binary_broadcast, dType)
}
void min_kernel(int dType, void *a, void *b, void *c, int a0, int a1, int a2,
                int a3, int b0, int b1, int b2, int b3, int c0, int c1, int c2,
                int c3) {

    int blocksize = block_work_size();
    int num = c0 * c1 * c2 * c3;
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    SWITCH_DTYPE(min, dType)
}
void max_kernel(int dType, void *a, void *b, void *c, int a0, int a1, int a2,
                int a3, int b0, int b1, int b2, int b3, int c0, int c1, int c2,
                int c3) {

    int blocksize = block_work_size();
    int num = c0 * c1 * c2 * c3;
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    SWITCH_DTYPE(max, dType)
}
void pow_kernel(int dType, void *a, void *b, void *c, int a0, int a1, int a2,
                int a3, int b0, int b1, int b2, int b3, int c0, int c1, int c2,
                int c3) {
    int blocksize = block_work_size();
    int num = c0 * c1 * c2 * c3;
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    if (dType == 1) {
        _pow_kernel<float>
            <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(
                a, b, c, a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3);
    } else if (dType == 3) {
        _pow_kernel<int8_t>
            <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(
                a, b, c, a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3);
    } else if (dType == 10) {
        _pow_kernel<half>
            <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(
                a, b, c, a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3);
    } else {
        IT_TODO_HALT();
    }
}
void less_kernel(int dType, void *a, void *b, void *c, int a0, int a1, int a2,
                 int a3, int b0, int b1, int b2, int b3, int c0, int c1, int c2,
                 int c3) {
    int blocksize = block_work_size();
    int num = c0 * c1 * c2 * c3;
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    SWITCH_DTYPE(less, dType)
}

}; // namespace infini
