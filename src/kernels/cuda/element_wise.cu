#include "cuda/cuda_common.h"
#include "cuda/cuda_utility.h"
#include <math.h>

constexpr unsigned int num_threads() { return 32 * 4; }
constexpr int thread_work_size() { return 4; }
constexpr int block_work_size() { return thread_work_size() * num_threads(); }

template <class T>
__global__ void _div_kernel(void *x, void *y, void *z, int a0, int a1,
                int a2, int a3, int a4, int b0, int b1, int b2, int b3, int b4,
                int c0, int c1, int c2, int c3, int c4) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int n = c0 * c1 * c2 * c3 * c4;

    for (int i = index; i < n; i += stride) {
        int c0_index = i / (c1 * c2 * c3 * c4);
        int remainder1 = i % (c1 * c2 * c3 * c4);
    
        int c1_index = remainder1 / (c2 * c3 * c4);
        int remainder2 = remainder1 % (c2 * c3 * c4);
    
        int c2_index = remainder2 / (c3 * c4);
        int remainder3 = remainder2 % (c3 * c4);
    
        int c3_index = remainder3 / c4;
        int c4_index = remainder3 % c4;

        int a0_index = c0_index % a0;
        int a1_index = c1_index % a1;
        int a2_index = c2_index % a2;
        int a3_index = c3_index % a3;
        int a4_index = c4_index % a4;

        int b0_index = c0_index % b0;
        int b1_index = c1_index % b1;
        int b2_index = c2_index % b2;
        int b3_index = c3_index % b3;
        int b4_index = c4_index % b4;
        if constexpr (std::is_same_v<T, half>) {
            // 对于FP16，使用FP32进行中间计算
            float x_val = __half2float(
                ((half *)x)[a0_index * a1 * a2 * a3 * a4 + a1_index * a2 * a3 * a4 +
                            a2_index * a3 * a4 + a3_index * a4 + a4_index]);
            float y_val = __half2float(
                ((half *)y)[b0_index * b1 * b2 * b3 * b4 + 
                       b1_index * b2 * b3 * b4 +
                       b2_index * b3 * b4 +
                       b3_index * b4 +
                       b4_index]);
            float result = x_val / y_val;
            ((half *)z)[i] = __float2half(result);
        } else {
            // 其他类型保持原逻辑
            ((T *)z)[i] =
                ((T *)x)[a0_index * a1 * a2 * a3 * a4 + 
                    a1_index * a2 * a3 * a4 +
                    a2_index * a3 * a4 +
                    a3_index * a4 +
                    a4_index] /
                ((T *)y)[b0_index * b1 * b2 * b3 * b4 + 
                    b1_index * b2 * b3 * b4 +
                    b2_index * b3 * b4 +
                    b3_index * b4 +
                    b4_index];
        }
    }
}

template <class T>
__global__ void _mul_kernel(void *x, void *y, void *z, int a0, int a1,
                int a2, int a3, int a4, int b0, int b1, int b2, int b3, int b4,
                int c0, int c1, int c2, int c3, int c4) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int n = c0 * c1 * c2 * c3 * c4;
    int a_s0 = a1 * a2 * a3 * a4;
    int a_s1 = a2 * a3 * a4;
    int a_s2 = a3 * a4;
    int a_s3 = a4;
    
    int b_s0 = b1 * b2 * b3 * b4;
    int b_s1 = b2 * b3 * b4;
    int b_s2 = b3 * b4;
    int b_s3 = b4;
    
    int c_s0 = c1 * c2 * c3 * c4;
    int c_s1 = c2 * c3 * c4;
    int c_s2 = c3 * c4;
    int c_s3 = c4;
    for (int i = index; i < n; i += stride) {
        int c0_index = i / c_s0;
        int c1_index = (i % c_s0) / c_s1;
        int c2_index = ((i % c_s0) % c_s1) / c_s2;
        int c3_index = (((i % c_s0) % c_s1) % c_s2) / c_s3;
        int c4_index = (((i % c_s0) % c_s1) % c_s2) % c_s3;
        int a0_index = c0_index % a0;
        int a1_index = c1_index % a1;
        int a2_index = c2_index % a2;
        int a3_index = c3_index % a3;
        int a4_index = c4_index % a4;
        int b0_index = c0_index % b0;
        int b1_index = c1_index % b1;
        int b2_index = c2_index % b2;
        int b3_index = c3_index % b3;
        int b4_index = c4_index % b4;
        ((T *)z)[i] = ((T *)x)[a0_index * a1 * a2 * a3 * a4 + 
                    a1_index * a2 * a3 * a4 +
                    a2_index * a3 * a4 +
                    a3_index * a4 +
                    a4_index] *
                      ((T *)y)[b0_index * b1 * b2 * b3 * b4 + 
                    b1_index * b2 * b3 * b4 +
                    b2_index * b3 * b4 +
                    b3_index * b4 +
                    b4_index];
    }
}

template <class T>
__global__ void _add_kernel(void *x, void *y, void *z, int a0, int a1,
                int a2, int a3, int a4, int b0, int b1, int b2, int b3, int b4,
                int c0, int c1, int c2, int c3, int c4) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int n = c0 * c1 * c2 * c3 * c4;

    for (int i = index; i < n; i += stride) {
        int c0_index = i / (c1 * c2 * c3 * c4);
        int remainder1 = i % (c1 * c2 * c3 * c4);
    
        int c1_index = remainder1 / (c2 * c3 * c4);
        int remainder2 = remainder1 % (c2 * c3 * c4);
    
        int c2_index = remainder2 / (c3 * c4);
        int remainder3 = remainder2 % (c3 * c4);
    
        int c3_index = remainder3 / c4;
        int c4_index = remainder3 % c4;

        int a0_index = c0_index % a0;
        int a1_index = c1_index % a1;
        int a2_index = c2_index % a2;
        int a3_index = c3_index % a3;
        int a4_index = c4_index % a4;

        int b0_index = c0_index % b0;
        int b1_index = c1_index % b1;
        int b2_index = c2_index % b2;
        int b3_index = c3_index % b3;
        int b4_index = c4_index % b4;
        ((T *)z)[i] = ((T *)x)[a0_index * a1 * a2 * a3 * a4 + 
                    a1_index * a2 * a3 * a4 +
                    a2_index * a3 * a4 +
                    a3_index * a4 +
                    a4_index] +
                      ((T *)y)[b0_index * b1 * b2 * b3 * b4 + 
                    b1_index * b2 * b3 * b4 +
                    b2_index * b3 * b4 +
                    b3_index * b4 +
                    b4_index];
    }
}
template <class T>
__global__ void _sub_kernel(void *x, void *y, void *z, int a0, int a1,
                int a2, int a3, int a4, int b0, int b1, int b2, int b3, int b4,
                int c0, int c1, int c2, int c3, int c4) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int n = c0 * c1 * c2 * c3 * c4;

    for (int i = index; i < n; i += stride) {
        int c0_index = i / (c1 * c2 * c3 * c4);
        int remainder1 = i % (c1 * c2 * c3 * c4);
    
        int c1_index = remainder1 / (c2 * c3 * c4);
        int remainder2 = remainder1 % (c2 * c3 * c4);
    
        int c2_index = remainder2 / (c3 * c4);
        int remainder3 = remainder2 % (c3 * c4);
    
        int c3_index = remainder3 / c4;
        int c4_index = remainder3 % c4;

        int a0_index = c0_index % a0;
        int a1_index = c1_index % a1;
        int a2_index = c2_index % a2;
        int a3_index = c3_index % a3;
        int a4_index = c4_index % a4;

        int b0_index = c0_index % b0;
        int b1_index = c1_index % b1;
        int b2_index = c2_index % b2;
        int b3_index = c3_index % b3;
        int b4_index = c4_index % b4;
        ((T *)z)[i] = ((T *)x)[a0_index * a1 * a2 * a3 * a4 + 
                    a1_index * a2 * a3 * a4 +
                    a2_index * a3 * a4 +
                    a3_index * a4 +
                    a4_index] -
                      ((T *)y)[b0_index * b1 * b2 * b3 * b4 + 
                    b1_index * b2 * b3 * b4 +
                    b2_index * b3 * b4 +
                    b3_index * b4 +
                    b4_index];
    }
}

template <class T>
__global__ void _pow_kernel(void *x, void *y, void *z, int a0, int a1,
                int a2, int a3, int a4, int b0, int b1, int b2, int b3, int b4,
                int c0, int c1, int c2, int c3, int c4) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int n = c0 * c1 * c2 * c3 * c4;

    for (int i = index; i < n; i += stride) {
        int c0_index = i / (c1 * c2 * c3 * c4);
        int remainder1 = i % (c1 * c2 * c3 * c4);
    
        int c1_index = remainder1 / (c2 * c3 * c4);
        int remainder2 = remainder1 % (c2 * c3 * c4);
    
        int c2_index = remainder2 / (c3 * c4);
        int remainder3 = remainder2 % (c3 * c4);
    
        int c3_index = remainder3 / c4;
        int c4_index = remainder3 % c4;

        int a0_index = c0_index % a0;
        int a1_index = c1_index % a1;
        int a2_index = c2_index % a2;
        int a3_index = c3_index % a3;
        int a4_index = c4_index % a4;

        int b0_index = c0_index % b0;
        int b1_index = c1_index % b1;
        int b2_index = c2_index % b2;
        int b3_index = c3_index % b3;
        int b4_index = c4_index % b4;
        ((T *)z)[i] =
            pow(((T *)x)[a0_index * a1 * a2 * a3 * a4 + 
                    a1_index * a2 * a3 * a4 +
                    a2_index * a3 * a4 +
                    a3_index * a4 +
                    a4_index],
                ((T *)y)[b0_index * b1 * b2 * b3 * b4 + 
                    b1_index * b2 * b3 * b4 +
                    b2_index * b3 * b4 +
                    b3_index * b4 +
                    b4_index]);
    }
}

template <class T>
__global__ void _less_kernel(void *x, void *y, void *z, int a0, int a1,
                int a2, int a3, int a4, int b0, int b1, int b2, int b3, int b4,
                int c0, int c1, int c2, int c3, int c4) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int n = c0 * c1 * c2 * c3 * c4;

    for (int i = index; i < n; i += stride) {
        int c0_index = i / (c1 * c2 * c3 * c4);
        int remainder1 = i % (c1 * c2 * c3 * c4);
    
        int c1_index = remainder1 / (c2 * c3 * c4);
        int remainder2 = remainder1 % (c2 * c3 * c4);
    
        int c2_index = remainder2 / (c3 * c4);
        int remainder3 = remainder2 % (c3 * c4);
    
        int c3_index = remainder3 / c4;
        int c4_index = remainder3 % c4;

        int a0_index = c0_index % a0;
        int a1_index = c1_index % a1;
        int a2_index = c2_index % a2;
        int a3_index = c3_index % a3;
        int a4_index = c4_index % a4;

        int b0_index = c0_index % b0;
        int b1_index = c1_index % b1;
        int b2_index = c2_index % b2;
        int b3_index = c3_index % b3;
        int b4_index = c4_index % b4;
        ((bool *)z)[i] =
            ((T *)x)[a0_index * a1 * a2 * a3 * a4 + 
                    a1_index * a2 * a3 * a4 +
                    a2_index * a3 * a4 +
                    a3_index * a4 +
                    a4_index] <
                    ((T *)y)[b0_index * b1 * b2 * b3 * b4 + 
                    b1_index * b2 * b3 * b4 +
                    b2_index * b3 * b4 +
                    b3_index * b4 +
                    b4_index]
                ? true
                : false;
    }
}
template <class T>
__global__ void _equal_kernel(void *x, void *y, void *z, int a0, int a1,
                int a2, int a3, int a4, int b0, int b1, int b2, int b3, int b4,
                int c0, int c1, int c2, int c3, int c4) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int n = c0 * c1 * c2 * c3 * c4;

    for (int i = index; i < n; i += stride) {
        int c0_index = i / (c1 * c2 * c3 * c4);
        int remainder1 = i % (c1 * c2 * c3 * c4);
    
        int c1_index = remainder1 / (c2 * c3 * c4);
        int remainder2 = remainder1 % (c2 * c3 * c4);
    
        int c2_index = remainder2 / (c3 * c4);
        int remainder3 = remainder2 % (c3 * c4);
    
        int c3_index = remainder3 / c4;
        int c4_index = remainder3 % c4;

        int a0_index = c0_index % a0;
        int a1_index = c1_index % a1;
        int a2_index = c2_index % a2;
        int a3_index = c3_index % a3;
        int a4_index = c4_index % a4;

        int b0_index = c0_index % b0;
        int b1_index = c1_index % b1;
        int b2_index = c2_index % b2;
        int b3_index = c3_index % b3;
        int b4_index = c4_index % b4;
        ((bool *)z)[i] =
            ((T *)x)[a0_index * a1 * a2 * a3 * a4 + 
                    a1_index * a2 * a3 * a4 +
                    a2_index * a3 * a4 +
                    a3_index * a4 +
                    a4_index] ==
                    ((T *)y)[b0_index * b1 * b2 * b3 * b4 + 
                    b1_index * b2 * b3 * b4 +
                    b2_index * b3 * b4 +
                    b3_index * b4 +
                    b4_index]
                ? true
                : false;
    }
}

#define CASE(OP, T)                                                            \
    _##OP##_kernel<DT_CUDA<T>::t>                                              \
        <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(          \
            a, b, c, a0, a1, a2, a3, a4, b0, b1, b2, b3, b4, c0, c1, c2, c3, c4);

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
        ((half *)z)[tid] = pow(((float)((half *)x)[tid]), *((half *)y));
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

void div_kernel(int dType, void *a, void *b, void *c, int a0, int a1,
                int a2, int a3, int a4, int b0, int b1, int b2, int b3, int b4, int c0,
                int c1, int c2, int c3, int c4) {

    int blocksize = block_work_size();
    int num = c0 * c1 * c2 * c3 * c4;
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    SWITCH_DTYPE(div, dType)
}
void mul_kernel(int dType, void *a, void *b, void *c, int a0, int a1,
                int a2, int a3, int a4, int b0, int b1, int b2, int b3, int b4, int c0,
                int c1, int c2, int c3, int c4) {

    int blocksize = block_work_size();
    int num = c0 * c1 * c2 * c3 * c4;
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    SWITCH_DTYPE(mul, dType)
}
void add_kernel(int dType, void *a, void *b, void *c, int a0, int a1,
                int a2, int a3, int a4, int b0, int b1, int b2, int b3, int b4, int c0,
                int c1, int c2, int c3, int c4) {

    int blocksize = block_work_size();
    int num = c0 * c1 * c2 * c3 * c4;
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    SWITCH_DTYPE(add, dType)
}
void sub_kernel(int dType, void *a, void *b, void *c, int a0, int a1,
                int a2, int a3, int a4, int b0, int b1, int b2, int b3, int b4, int c0,
                int c1, int c2, int c3, int c4) {

    int blocksize = block_work_size();
    int num = c0 * c1 * c2 * c3 * c4;
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    SWITCH_DTYPE(sub, dType)
}
void pow_kernel(int dType, void *a, void *b, void *c, int a0, int a1,
                int a2, int a3, int a4, int b0, int b1, int b2, int b3, int b4, int c0,
                int c1, int c2, int c3, int c4) {
    int blocksize = block_work_size();
    int num = c0 * c1 * c2 * c3 * c4;
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    if (dType == 1) {
        _pow_kernel<float>
            <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(
                a, b, c, a0, a1, a2, a3, a4, b0, b1, b2, b3, b4, c0, c1, c2, c3, c4);
    } else if (dType == 3) {
        _pow_kernel<int8_t>
            <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(
                a, b, c, a0, a1, a2, a3, a4, b0, b1, b2, b3, b4, c0, c1, c2, c3, c4);
    } else if (dType == 10) {
        int a_size = a0 * a1 * a2 * a3 * a4;
        int b_size = b0 * b1 * b2 * b3 * b4;
        int c_size = c0 * c1 * c2 * c3 * c4;
        vector<float> a_float(a_size);
        vector<float> b_float(b_size);
        vector<float> c_float(c_size);
        for (int i = 0; i < a_size; ++i) {
            a_float[i] = __half2float(((half *)a)[i]);
        }
        for (int i = 0; i < b_size; ++i) {
            b_float[i] = __half2float(((half *)b)[i]);
        }
        _pow_kernel<float>
            <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(
                a_float.data(), b_float.data(), c_float.data(), a0, a1, a2, a3, a4, b0, b1, b2, b3, b4, c0, c1, c2, c3, c4);
        for (int i = 0; i < c_size; ++i) {
            ((half *)c)[i] = __float2half(c_float[i]);
        }
    } else {
        IT_TODO_HALT();
    }
}
void less_kernel(int dType, void *a, void *b, void *c, int a0, int a1,
                int a2, int a3, int a4, int b0, int b1, int b2, int b3, int b4,
                int c0, int c1, int c2, int c3, int c4) {
    int blocksize = block_work_size();
    int num = c0 * c1 * c2 * c3 * c4;
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    SWITCH_DTYPE(less, dType)
}
void equal_kernel(int dType, void *a, void *b, void *c, int a0, int a1,
                int a2, int a3, int a4, int b0, int b1, int b2, int b3, int b4,
                int c0, int c1, int c2, int c3, int c4) {
    int blocksize = block_work_size();
    int num = c0 * c1 * c2 * c3 * c4;
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    SWITCH_DTYPE(equal, dType)
}

}; // namespace infini
