#include "core/common.h"
#include "cuda/cuda_common.h"
#include "cuda/cuda_utility.h"
#include "utils/small_array.h"

constexpr unsigned int num_threads() { return 32 * 4; }
constexpr int thread_work_size() { return 4; }
constexpr int block_work_size() { return thread_work_size() * num_threads(); }
const int repeat = 1;
template <class T>
__global__ void _expandKernel(void *input, void *output, int a0, int a1, int a2,
                              int a3, int b0, int b1, int b2, int b3) {

    int index = threadIdx.x + blockIdx.x * blockDim.x;

    int stride1 = b2 * b3;
    int stride0 = b1 * stride1;
    int n = b0 * stride0;
    int end = (repeat * index + repeat < n ? repeat * index + repeat : n);
    for (int i = repeat * index; i < end; i++) {
        int xIdx = (a0 * a1 * a2 * a3 == n ? i : 0);
        bool aIdx = (a0 * a1 * a2 * a3 < n && a0 * a1 * a2 * a3 > 1);
        if (aIdx) {
            int b0_index = i / stride0;
            int b1_index = (i % stride0) / stride1;
            int b2_index = (i % stride1) / b3;
            int b3_index = i % b3;
            xIdx = (b0_index % a0) * a1 * a2 * a3 + (b1_index % a1) * a2 * a3 +
                   (b2_index % a2) * a3 + b3_index % a3;
        }
        ((T *)output)[i] = ((T *)input)[xIdx];
    }
}
template <class T>
__global__ void _expandKernel(void *input, void *output, int nDims,
                              int outputsize, infini::SmallArray inputShape,
                              infini::SmallArray outputShape) {

    int outputIdx =
        blockIdx.x * blockDim.x + threadIdx.x; // i(JKS) + j(KS) + k(S) + s
    if (outputIdx < outputsize) {
        int inputIdx = 0;  // record input index
        int temp = 1;      // stored S, KS, JKS, in order
        int tmp = 1;       // stored s,k,j,i in order
        int v = outputIdx; // v = i(JKS) + j(KS) + k(S) + s
        for (int i = nDims - 1; i >= 0; --i) {
            if (i == 0) {
                tmp = v; // i = outputIdx/(JKS)
            } else {
                tmp = v % outputShape.data[i]; // store s,k,j in order
            }
            if (inputShape.data[i] ==
                1) { // if input shape = 1, the index only equal 0
                inputIdx += 0;
            } else {
                inputIdx +=
                    tmp * temp; // otherwise +i(JKS) or j(KS) or k(S) or s
            }
            temp *= inputShape.data[i];
            v = v / outputShape.data[i];
        }
        ((T *)output)[outputIdx] = ((T *)input)[inputIdx];
    }
}
template <class T>
static __global__ void _expandRowKernel(void *__restrict__ dst,
                                        void const *__restrict__ src) {
    auto da = gridDim.x, db = blockDim.y, dx = blockDim.x, n = blockIdx.y,
         a = blockIdx.x, b = threadIdx.y, x = threadIdx.x;
    auto i = ((n * da + a) * db + b) * dx + x, j = (a * db + b) * dx + x;
    reinterpret_cast<T *>(dst)[i] = reinterpret_cast<T const *>(src)[j];
}
namespace infini {

#define CASE(T)                                                                \
    _expandKernel<DT_CUDA<T>::t>                                               \
        <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(          \
            input, output, a0, a1, a2, a3, b0, b1, b2, b3);

#define SWITCH_DTYPE(DTYPE)                                                    \
    switch (DTYPE) {                                                           \
    case 1:                                                                    \
        CASE(1)                                                                \
        break;                                                                 \
    case 2:                                                                    \
        CASE(2)                                                                \
        break;                                                                 \
    case 3:                                                                    \
        CASE(3)                                                                \
        break;                                                                 \
    case 4:                                                                    \
        CASE(4)                                                                \
        break;                                                                 \
    case 5:                                                                    \
        CASE(5)                                                                \
        break;                                                                 \
    case 6:                                                                    \
        CASE(6)                                                                \
        break;                                                                 \
    case 7:                                                                    \
        CASE(7)                                                                \
        break;                                                                 \
    case 10:                                                                   \
        CASE(10)                                                               \
        break;                                                                 \
    case 11:                                                                   \
        CASE(11)                                                               \
        break;                                                                 \
    case 12:                                                                   \
        CASE(12)                                                               \
        break;                                                                 \
    case 13:                                                                   \
        CASE(13)                                                               \
        break;                                                                 \
    case 16:                                                                   \
        CASE(16)                                                               \
        break;                                                                 \
    default:                                                                   \
        IT_TODO_HALT();                                                        \
    }

void expandKernel(int dType, void *input, void *output, int a0, int a1, int a2,
                  int a3, int b0, int b1, int b2, int b3) {
    int blocksize = block_work_size();
    int outputsize = b0 * b1 * b2 * b3;
    int gridsize = (outputsize + repeat * block_work_size() - 1) /
                   (repeat * block_work_size());
    SWITCH_DTYPE(dType)
}
#define CASECurrency(T)                                                        \
    _expandKernel<DT_CUDA<T>::t>                                               \
        <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(          \
            input, output, nDims, outputsize, inputShape, outputShape);

#define SWITCHCurrency_DTYPE(DTYPE)                                            \
    switch (DTYPE) {                                                           \
    case 1:                                                                    \
        CASECurrency(1) break;                                                 \
    case 2:                                                                    \
        CASECurrency(2) break;                                                 \
    case 3:                                                                    \
        CASECurrency(3) break;                                                 \
    case 4:                                                                    \
        CASECurrency(4) break;                                                 \
    case 5:                                                                    \
        CASECurrency(5) break;                                                 \
    case 6:                                                                    \
        CASECurrency(6) break;                                                 \
    case 7:                                                                    \
        CASECurrency(7) break;                                                 \
    case 10:                                                                   \
        CASECurrency(10) break;                                                \
    case 11:                                                                   \
        CASECurrency(11) break;                                                \
    case 12:                                                                   \
        CASECurrency(12) break;                                                \
    case 13:                                                                   \
        CASECurrency(13) break;                                                \
    case 16:                                                                   \
        CASECurrency(16) break;                                                \
    default:                                                                   \
        IT_TODO_HALT();                                                        \
    }

void expandKernel(int dType, void *input, void *output, int nDims,
                  int outputsize, SmallArray inputShape,
                  SmallArray outputShape) {
    int blocksize = block_work_size();
    int gridsize = (outputsize + block_work_size() - 1) / block_work_size();
    SWITCHCurrency_DTYPE(dType)
}
#define CASE_ROW(T)                                                            \
    _expandRowKernel<float>                                                    \
        <<<grid, block, 0, CUDAStream::getCurrentStream()>>>(output, input);

#define SWITCH_DTYPE_ROW(DTYPE)                                                \
    switch (DTYPE) {                                                           \
    case 1:                                                                    \
        CASE_ROW(1)                                                            \
        break;                                                                 \
    case 2:                                                                    \
        CASE_ROW(2)                                                            \
        break;                                                                 \
    case 3:                                                                    \
        CASE_ROW(3)                                                            \
        break;                                                                 \
    case 4:                                                                    \
        CASE_ROW(4)                                                            \
        break;                                                                 \
    case 5:                                                                    \
        CASE_ROW(5)                                                            \
        break;                                                                 \
    case 6:                                                                    \
        CASE_ROW(6)                                                            \
        break;                                                                 \
    case 7:                                                                    \
        CASE_ROW(7)                                                            \
        break;                                                                 \
    case 10:                                                                   \
        CASE_ROW(10)                                                           \
        break;                                                                 \
    case 11:                                                                   \
        CASE_ROW(11)                                                           \
        break;                                                                 \
    case 12:                                                                   \
        CASE_ROW(12)                                                           \
        break;                                                                 \
    case 13:                                                                   \
        CASE_ROW(13)                                                           \
        break;                                                                 \
    case 16:                                                                   \
        CASE_ROW(16)                                                           \
        break;                                                                 \
    default:                                                                   \
        IT_TODO_HALT();                                                        \
    }

// Optimization for expanding a row vector. The row length must be a multiple of
// 32
void expandRowKernel(int dType, void *input, void *output, int n_rows,
                     int row_len) {
    // Factorize row_len: row_len = a x b x 32 (32 is the warp size), b<=32
    // input: 1 x (a x b x 32 x sizeT)
    // output: n_rows x (a x b x 32 x sizeT)
    // grid: n_rows x a
    // block: b x 32
    auto c = row_len / 32, b = c;
    if (b > 32) {
        for (b = 32; c % b != 0; --b)
            ;
    }
    auto a = c / b;
    dim3 grid(a, n_rows), block(32, b);
    SWITCH_DTYPE_ROW(dType)
}

} // namespace infini
