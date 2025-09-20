#include "cuda/cuda_common.h"
#include "cuda/cuda_utility.h"
#include <cassert>
#include <cuda_runtime.h>
#include <math.h>
#include <type_traits>

constexpr unsigned int num_threads() { return 32 * 4; }
constexpr int thread_work_size() { return 4; }
constexpr int block_work_size() { return thread_work_size() * num_threads(); }

namespace infini {
// Device implementations for logical and bitwise operators.
// Templates below implement element-wise kernels for binary/unary ops and
// several small functors (AndOp, OrOp, BitAndOp, ...). These kernels are
// instantiated and launched by the host-side functions declared in
// `include/cuda/cuda_logical.h`.

template <typename T, typename Op>
__global__ void _logical_binary_kernel(void *a, void *b, void *c, int a0,
                                       int a1, int a2, int a3, int b0, int b1,
                                       int b2, int b3, int c0, int c1, int c2,
                                       int c3) {

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int n = c0 * c1 * c2 * c3;

    T *__restrict__ Ap = reinterpret_cast<T *>(a);
    T *__restrict__ Bp = reinterpret_cast<T *>(b);
    T *__restrict__ Cp = reinterpret_cast<T *>(c);

    int a123 = a1 * a2 * a3;
    int a23 = a2 * a3;
    int a3_ = a3;

    int b123 = b1 * b2 * b3;
    int b23 = b2 * b3;
    int b3_ = b3;

    int c123 = c1 * c2 * c3;
    int c23 = c2 * c3;
    int c3_ = c3;

    for (int i = index; i < n; i += stride) {
        int c0i = i / c123;
        int r1 = i - c0i * c123;
        int c1i = r1 / c23;
        int r2 = r1 - c1i * c23;
        int c2i = r2 / c3_;
        int c3i = r2 - c2i * c3_;

        int a0i = (a0 == 1) ? 0 : (c0i % a0);
        int a1i = (a1 == 1) ? 0 : (c1i % a1);
        int a2i = (a2 == 1) ? 0 : (c2i % a2);
        int a3i = (a3 == 1) ? 0 : (c3i % a3);

        int b0i = (b0 == 1) ? 0 : (c0i % b0);
        int b1i = (b1 == 1) ? 0 : (c1i % b1);
        int b2i = (b2 == 1) ? 0 : (c2i % b2);
        int b3i = (b3 == 1) ? 0 : (c3i % b3);

        int aoff = ((a0i * a1 + a1i) * a2 + a2i) * a3 + a3i;
        int boff = ((b0i * b1 + b1i) * b2 + b2i) * b3 + b3i;

        T va = Ap[aoff];
        T vb = Bp[boff];

        Cp[i] = Op::template apply<T>(va, vb);
    }
}

template <typename T, typename Op>
__global__ void _logical_unary_kernel(void *a, void *b, int a0, int a1, int a2,
                                      int a3, int b0, int b1, int b2, int b3) {

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int n = b0 * b1 * b2 * b3;

    T *__restrict__ Ap = reinterpret_cast<T *>(a);
    T *__restrict__ Bp = reinterpret_cast<T *>(b);

    int a123 = a1 * a2 * a3;
    int a23 = a2 * a3;
    int a3_ = a3;

    int b123 = b1 * b2 * b3;
    int b23 = b2 * b3;
    int b3_ = b3;

    for (int i = index; i < n; i += stride) {
        int b0i = i / b123;
        int r1 = i - b0i * b123;
        int b1i = r1 / b23;
        int r2 = r1 - b1i * b23;
        int b2i = r2 / b3_;
        int b3i = r2 - b2i * b3_;

        int a0i = (a0 == 1) ? 0 : (b0i % a0);
        int a1i = (a1 == 1) ? 0 : (b1i % a1);
        int a2i = (a2 == 1) ? 0 : (b2i % a2);
        int a3i = (a3 == 1) ? 0 : (b3i % a3);

        int aoff = ((a0i * a1 + a1i) * a2 + a2i) * a3 + a3i;

        T va = Ap[aoff];

        Bp[i] = Op::template apply<T>(va);
    }
}

struct AndOp {
    template <typename T> static __device__ T apply(T a, T b) { return a && b; }
};

struct OrOp {
    template <typename T> static __device__ T apply(T a, T b) { return a || b; }
};

struct XorOp {
    template <typename T> static __device__ T apply(T a, T b) { return a != b; }
};

struct NotOp {
    template <typename T> static __device__ T apply(T a) { return !a; }
};

struct BitAndOp {
    template <typename T> static __device__ T apply(T a, T b) { return a & b; }
};

struct BitOrOp {
    template <typename T> static __device__ T apply(T a, T b) { return a | b; }
};

struct BitXorOp {
    template <typename T> static __device__ T apply(T a, T b) { return a ^ b; }
};

struct BitNotOp {
    template <typename T> static __device__ T apply(T a) { return ~a; }
};

struct BitLeftShiftOp {
    template <typename T> static __device__ T apply(T a, T b) { return a << b; }
};

struct BitRightShiftOp {
    template <typename T> static __device__ T apply(T a, T b) { return a >> b; }
};

// -----------------------------------------------------------------------------
// 1=bool, 3=uchar, 4=char, 5=ushort, 6=short, 7=int, 9=bool, 12=uint, 13=ull
// -----------------------------------------------------------------------------
// 内核 launch
#define BINARY_CASE(OP, T)                                                     \
    \ 
    _logical_binary_kernel<DT_CUDA<T>::t, OP>                                  \
        <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(          \
            a, b, c, a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3);

#define UNARY_CASE(OP, T)                                                      \
    \ 
    _logical_unary_kernel<DT_CUDA<T>::t, OP>                                   \
        <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(          \
            a, b, a0, a1, a2, a3, b0, b1, b2, b3);

#define SWITCH_DTYPE_BINARY_BOOL(OP, DTYPE)                                    \
    switch (DTYPE) {                                                           \
    case 9:                                                                    \
        BINARY_CASE(OP, 9);                                                    \
        break;                                                                 \
    default:                                                                   \
        IT_TODO_HALT();                                                        \
    }

#define SWITCH_DTYPE_UNARY_BOOL(OP, DTYPE)                                     \
    switch (DTYPE) {                                                           \
    case 9:                                                                    \
        UNARY_CASE(OP, 9);                                                     \
        break;                                                                 \
    default:                                                                   \
        IT_TODO_HALT();                                                        \
    }

#define SWITCH_DTYPE_BINARY_INT(OP, DTYPE)                                     \
    switch (DTYPE) {                                                           \
    case 2:                                                                    \
        BINARY_CASE(OP, 2)                                                     \
        break;                                                                 \
    case 3:                                                                    \
        BINARY_CASE(OP, 3)                                                     \
        break;                                                                 \
    case 4:                                                                    \
        BINARY_CASE(OP, 4)                                                     \
        break;                                                                 \
    case 5:                                                                    \
        BINARY_CASE(OP, 5)                                                     \
        break;                                                                 \
    case 6:                                                                    \
        BINARY_CASE(OP, 6)                                                     \
        break;                                                                 \
    case 7:                                                                    \
        BINARY_CASE(OP, 7)                                                     \
        break;                                                                 \
    case 12:                                                                   \
        BINARY_CASE(OP, 12)                                                    \
        break;                                                                 \
    case 13:                                                                   \
        BINARY_CASE(OP, 13)                                                    \
        break;                                                                 \
    default:                                                                   \
        IT_TODO_HALT();                                                        \
    }

#define SWITCH_DTYPE_UNARY_INT(OP, DTYPE)                                      \
    switch (DTYPE) {                                                           \
    case 2:                                                                    \
        UNARY_CASE(OP, 2)                                                      \
        break;                                                                 \
    case 3:                                                                    \
        UNARY_CASE(OP, 3)                                                      \
        break;                                                                 \
    case 4:                                                                    \
        UNARY_CASE(OP, 4)                                                      \
        break;                                                                 \
    case 5:                                                                    \
        UNARY_CASE(OP, 5)                                                      \
        break;                                                                 \
    case 6:                                                                    \
        UNARY_CASE(OP, 6)                                                      \
        break;                                                                 \
    case 7:                                                                    \
        UNARY_CASE(OP, 7)                                                      \
        break;                                                                 \
    case 12:                                                                   \
        UNARY_CASE(OP, 12)                                                     \
        break;                                                                 \
    case 13:                                                                   \
        UNARY_CASE(OP, 13)                                                     \
        break;                                                                 \
    default:                                                                   \
        IT_TODO_HALT();                                                        \
    }

inline void _compute_grid_block(int total_elems, int &gridsize,
                                int &blocksize) {
    blocksize = block_work_size();
    gridsize = (total_elems + blocksize - 1) / blocksize;
}

void And_kernel(int dtypeIndex, void *a, void *b, void *c, int a0, int a1,
                int a2, int a3, int b0, int b1, int b2, int b3, int c0, int c1,
                int c2, int c3) {
    int blocksize, gridsize;
    _compute_grid_block(c0 * c1 * c2 * c3, gridsize, blocksize);
    SWITCH_DTYPE_BINARY_BOOL(AndOp, dtypeIndex)
}

void Or_kernel(int dtypeIndex, void *a, void *b, void *c, int a0, int a1,
               int a2, int a3, int b0, int b1, int b2, int b3, int c0, int c1,
               int c2, int c3) {
    int blocksize, gridsize;
    _compute_grid_block(c0 * c1 * c2 * c3, gridsize, blocksize);
    SWITCH_DTYPE_BINARY_BOOL(OrOp, dtypeIndex)
}

void Xor_kernel(int dtypeIndex, void *a, void *b, void *c, int a0, int a1,
                int a2, int a3, int b0, int b1, int b2, int b3, int c0, int c1,
                int c2, int c3) {
    int blocksize, gridsize;
    _compute_grid_block(c0 * c1 * c2 * c3, gridsize, blocksize);
    printf("Xor_kernel dtypeIndex=%d\n", dtypeIndex);
    SWITCH_DTYPE_BINARY_BOOL(XorOp, dtypeIndex)
}

void Not_kernel(int dtypeIndex, void *a, void *b, int a0, int a1, int a2,
                int a3, int b0, int b1, int b2, int b3) {
    int blocksize, gridsize;
    _compute_grid_block(b0 * b1 * b2 * b3, gridsize, blocksize);
    SWITCH_DTYPE_UNARY_BOOL(NotOp, dtypeIndex)
}

void BitAnd_kernel(int dtypeIndex, void *a, void *b, void *c, int a0, int a1,
                   int a2, int a3, int b0, int b1, int b2, int b3, int c0,
                   int c1, int c2, int c3) {
    int blocksize, gridsize;
    _compute_grid_block(c0 * c1 * c2 * c3, gridsize, blocksize);
    SWITCH_DTYPE_BINARY_INT(BitAndOp, dtypeIndex)
}

void BitOr_kernel(int dtypeIndex, void *a, void *b, void *c, int a0, int a1,
                  int a2, int a3, int b0, int b1, int b2, int b3, int c0,
                  int c1, int c2, int c3) {
    int blocksize, gridsize;
    _compute_grid_block(c0 * c1 * c2 * c3, gridsize, blocksize);
    SWITCH_DTYPE_BINARY_INT(BitOrOp, dtypeIndex)
}

void BitXor_kernel(int dtypeIndex, void *a, void *b, void *c, int a0, int a1,
                   int a2, int a3, int b0, int b1, int b2, int b3, int c0,
                   int c1, int c2, int c3) {
    int blocksize, gridsize;
    _compute_grid_block(c0 * c1 * c2 * c3, gridsize, blocksize);
    SWITCH_DTYPE_BINARY_INT(BitXorOp, dtypeIndex)
}

void BitNot_kernel(int dtypeIndex, void *a, void *b, int a0, int a1, int a2,
                   int a3, int b0, int b1, int b2, int b3) {
    int blocksize, gridsize;
    _compute_grid_block(b0 * b1 * b2 * b3, gridsize, blocksize);
    SWITCH_DTYPE_UNARY_INT(BitNotOp, dtypeIndex)
}

void BitLeftShift_kernel(int dtypeIndex, void *a, void *b, void *c, int a0,
                         int a1, int a2, int a3, int b0, int b1, int b2, int b3,
                         int c0, int c1, int c2, int c3) {
    int blocksize, gridsize;
    _compute_grid_block(c0 * c1 * c2 * c3, gridsize, blocksize);
    SWITCH_DTYPE_BINARY_INT(BitLeftShiftOp, dtypeIndex)
}

void BitRightShift_kernel(int dtypeIndex, void *a, void *b, void *c, int a0,
                          int a1, int a2, int a3, int b0, int b1, int b2,
                          int b3, int c0, int c1, int c2, int c3) {
    int blocksize, gridsize;
    _compute_grid_block(c0 * c1 * c2 * c3, gridsize, blocksize);
    SWITCH_DTYPE_BINARY_INT(BitRightShiftOp, dtypeIndex)
}

} // namespace infini
