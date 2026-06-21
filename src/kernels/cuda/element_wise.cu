#include "cuda/cuda_common.h"
#include "cuda/cuda_utility.h"
#include <math.h>

constexpr unsigned int num_threads() { return 32 * 4; }
constexpr int thread_work_size() { return 4; }
constexpr int block_work_size() { return thread_work_size() * num_threads(); }

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
__global__ void _mul_kernel(void *x, void *y, void *z, int a0, int a1, int a2,
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
                               a2_index * a3 + a3_index] *
                      ((T *)y)[b0_index * b1 * b2 * b3 + b1_index * b2 * b3 +
                               b2_index * b3 + b3_index];
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
        ((T *)z)[i] =
            pow(((T *)x)[a0_index * a1 * a2 * a3 + a1_index * a2 * a3 +
                         a2_index * a3 + a3_index],
                ((T *)y)[b0_index * b1 * b2 * b3 + b1_index * b2 * b3 +
                         b2_index * b3 + b3_index]);
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
template <class T>
__global__ void _equal_kernel(void *x, void *y, void *z, int a0, int a1, int a2,
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
                     a2_index * a3 + a3_index] ==
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
__global__ void _div_special_kernel(void *x, void *y, void *z, int num) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num; i += stride) {
        ((T *)z)[i] = ((T *)x)[i] / ((T *)y)[i];
    }
}
template <class T>
__global__ void _add_special_kernel(void *x, void *y, void *z, int num) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num; i += stride) {
        ((T *)z)[i] = ((T *)x)[i] + ((T *)y)[i];
    }
}
template <class T>
__global__ void _mul_special_kernel(void *x, void *y, void *z, int num) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num; i += stride) {
        ((T *)z)[i] = ((T *)x)[i] * ((T *)y)[i];
    }
}
template <class T>
__global__ void _pow_special_kernel(void *x, void *y, void *z, int num) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num; i += stride) {
        ((T *)z)[i] = pow(((T *)x)[i], ((T *)y)[i]);
    }
}
template <class T>
__global__ void _less_special_kernel(void *x, void *y, void *z, int num) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num; i += stride) {
        ((bool *)z)[i] = ((T *)x)[i] < ((T *)y)[i] ? true : false;
    }
}
template <class T>
__global__ void _equal_special_kernel(void *x, void *y, void *z, int num) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num; i += stride) {
        ((bool *)z)[i] = ((T *)x)[i] == ((T *)y)[i] ? true : false;
    }
}
#define CASE_SPECIAL(OP, T)                                                    \
    _##OP##_special_kernel<DT_CUDA<T>::t>                                      \
        <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(a, b, c,  \
                                                                     num);

#define SWITCH_DTYPE_SPECIAL(OP, DTYPE)                                        \
    switch (DTYPE) {                                                           \
    case 1:                                                                    \
        CASE_SPECIAL(OP, 1)                                                    \
        break;                                                                 \
    case 2:                                                                    \
        CASE_SPECIAL(OP, 2)                                                    \
        break;                                                                 \
    case 3:                                                                    \
        CASE_SPECIAL(OP, 3)                                                    \
        break;                                                                 \
    case 4:                                                                    \
        CASE_SPECIAL(OP, 4)                                                    \
        break;                                                                 \
    case 5:                                                                    \
        CASE_SPECIAL(OP, 5)                                                    \
        break;                                                                 \
    case 6:                                                                    \
        CASE_SPECIAL(OP, 6)                                                    \
        break;                                                                 \
    case 7:                                                                    \
        CASE_SPECIAL(OP, 7)                                                    \
        break;                                                                 \
    case 10:                                                                   \
        CASE_SPECIAL(OP, 10)                                                   \
        break;                                                                 \
    case 11:                                                                   \
        CASE_SPECIAL(OP, 11)                                                   \
        break;                                                                 \
    case 12:                                                                   \
        CASE_SPECIAL(OP, 12)                                                   \
        break;                                                                 \
    case 13:                                                                   \
        CASE_SPECIAL(OP, 13)                                                   \
        break;                                                                 \
    case 16:                                                                   \
        CASE_SPECIAL(OP, 16)                                                   \
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
__global__ void _mul_const_kernel(void const *__restrict__ x,
                                  void const *__restrict__ y,
                                  void *__restrict__ z, const size_t n) {
    T val = *((T *)y); // 主线程提前读一次
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        ((T *)z)[tid] = ((T *)x)[tid] * val;
    }
}

template <class T>
__global__ void _add_const_kernel(void const *__restrict__ x,
                                  void const *__restrict__ y,
                                  void *__restrict__ z, const size_t n) {
    T val = *((T *)y); // 主线程提前读一次
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        ((T *)z)[tid] = ((T *)x)[tid] + val;
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

void mul_const_kernel(int dType, void *a, void *b, void *c, size_t n) {
    size_t blocksize = block_work_size();
    size_t gridsize = (n + block_work_size() - 1) / block_work_size();
    SWITCH_DTYPE_CONST(mul, dType);
}

void add_const_kernel(int dType, void *a, void *b, void *c, size_t n) {
    size_t blocksize = block_work_size();
    size_t gridsize = (n + block_work_size() - 1) / block_work_size();
    SWITCH_DTYPE_CONST(add, dType);
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

    int blocksize = block_work_size();
    int num = c0 * c1 * c2 * c3;
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    SWITCH_DTYPE(add, dType)
}
void mul_kernel(int dType, void *a, void *b, void *c, int a0, int a1, int a2,
                int a3, int b0, int b1, int b2, int b3, int c0, int c1, int c2,
                int c3) {

    int blocksize = block_work_size();
    int num = c0 * c1 * c2 * c3;
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    SWITCH_DTYPE(mul, dType)
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
        int a_size = a0 * a1 * a2 * a3;
        int b_size = b0 * b1 * b2 * b3;
        int c_size = c0 * c1 * c2 * c3;
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
                a_float.data(), b_float.data(), c_float.data(), a0, a1, a2, a3,
                b0, b1, b2, b3, c0, c1, c2, c3);
        for (int i = 0; i < c_size; ++i) {
            ((half *)c)[i] = __float2half(c_float[i]);
        }
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
void equal_kernel(int dType, void *a, void *b, void *c, int a0, int a1, int a2,
                  int a3, int b0, int b1, int b2, int b3, int c0, int c1,
                  int c2, int c3) {
    int blocksize = block_work_size();
    int num = c0 * c1 * c2 * c3;
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    SWITCH_DTYPE(equal, dType)
}

void div_special_kernel(int dType, void *a, void *b, void *c, int num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    SWITCH_DTYPE_SPECIAL(div, dType)
}
void add_special_kernel(int dType, void *a, void *b, void *c, int num) {
    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    SWITCH_DTYPE_SPECIAL(add, dType)
}
void mul_special_kernel(int dType, void *a, void *b, void *c, int num) {
    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    SWITCH_DTYPE_SPECIAL(mul, dType)
}
void pow_special_kernel(int dType, void *a, void *b, void *c, int num) {
    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    if (dType == 1) {
        _pow_special_kernel<float>
            <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(
                a, b, c, num);
    } else if (dType == 3) {
        _pow_special_kernel<int8_t>
            <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(
                a, b, c, num);
    } else if (dType == 10) {
        vector<float> a_float(num);
        vector<float> b_float(num);
        vector<float> c_float(num);
        for (int i = 0; i < num; ++i) {
            a_float[i] = __half2float(((half *)a)[i]);
            b_float[i] = __half2float(((half *)b)[i]);
        }
        _pow_special_kernel<float>
            <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(
                a_float.data(), b_float.data(), c_float.data(), num);
        for (int i = 0; i < num; ++i) {
            ((half *)c)[i] = __float2half(c_float[i]);
        }
    } else {
        IT_TODO_HALT();
    }
}
void less_special_kernel(int dType, void *a, void *b, void *c, int num) {
    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    SWITCH_DTYPE_SPECIAL(less, dType)
}
void equal_special_kernel(int dType, void *a, void *b, void *c, int num) {
    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    SWITCH_DTYPE_SPECIAL(equal, dType)
}

}; // namespace infini
