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

#define CASE(OP, T)                                                            \
    _##OP##_kernel<DT_CUDA<T>::t><<<gridsize, blocksize, 0, CUDAStream::stream>>>(                    \
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

namespace infini {
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
void pow_kernel(int dType, void *a, void *b, void *c, int a0, int a1, int a2,
                int a3, int b0, int b1, int b2, int b3, int c0, int c1, int c2,
                int c3) {
    int blocksize = block_work_size();
    int num = c0 * c1 * c2 * c3;
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    if (dType == 1) {
        _pow_kernel<float><<<gridsize, blocksize, 0, CUDAStream::stream>>>(a, b, c, a0, a1, a2, a3, b0,
                                                    b1, b2, b3, c0, c1, c2, c3);
    } else if (dType == 3) {
        _pow_kernel<int8_t><<<gridsize, blocksize, 0, CUDAStream::stream>>>(
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
        _pow_kernel<float><<<gridsize, blocksize, 0, CUDAStream::stream>>>(
            a_float.data(), b_float.data(), c_float.data(), a0, a1, a2, a3, b0,
            b1, b2, b3, c0, c1, c2, c3);
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

}; // namespace infini
