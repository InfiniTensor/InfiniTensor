#include "core/common.h"
#include "cuda/cuda_common.h"
#include "cuda/cuda_utility.h"

constexpr unsigned int num_threads() { return 32 * 4; }
constexpr int thread_work_size() { return 4; }
constexpr int block_work_size() { return thread_work_size() * num_threads(); }
const int repeat = 3;
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

} // namespace infini
