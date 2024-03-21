#include "cuda/cuda_common.h"
#include "cuda/cuda_utility.h"

template <typename T>
__global__ void
_whereKernel(void *inputX, void *inputY, const uint8_t *condition, void *output,
             int a0, int a1, int a2, int a3, int b0, int b1, int b2, int b3,
             int c0, int c1, int c2, int c3, int d0, int d1, int d2, int d3) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int outputsize = d0 * d1 * d2 * d3;
    if (i < outputsize) {

        int d0_index = i / (d1 * d2 * d3);
        int d1_index = (i % (d1 * d2 * d3)) / (d2 * d3);
        int d2_index = ((i % (d1 * d2 * d3)) % (d2 * d3)) / d3;
        int d3_index = ((i % (d1 * d2 * d3)) % (d2 * d3)) % d3;

        int a0_index = d0_index % a0;
        int a1_index = d1_index % a1;
        int a2_index = d2_index % a2;
        int a3_index = d3_index % a3;

        int b0_index = d0_index % b0;
        int b1_index = d1_index % b1;
        int b2_index = d2_index % b2;
        int b3_index = d3_index % b3;

        int c0_index = d0_index % c0;
        int c1_index = d1_index % c1;
        int c2_index = d2_index % c2;
        int c3_index = d3_index % c3;

        int inputXIdx = a0_index * a1 * a2 * a3 + a1_index * a2 * a3 +
                        a2_index * a3 + a3_index;
        int inputYIdx = b0_index * b1 * b2 * b3 + b1_index * b2 * b3 +
                        b2_index * b3 + b3_index;
        int conditionIdx = c0_index * c1 * c2 * c3 + c1_index * c2 * c3 +
                           c2_index * c3 + c3_index;
        ((T *)output)[i] = condition[conditionIdx] ? ((T *)inputX)[inputXIdx]
                                                   : ((T *)inputY)[inputYIdx];
    }
}
#define CASE(T)                                                                \
    _whereKernel<DT_CUDA<T>::t>                                                \
        <<<gridsize, blocksize, 0, CUDAStream::stream>>>(                      \
            inputX, inputY, condition, output, a0, a1, a2, a3, b0, b1, b2, b3, \
            c0, c1, c2, c3, d0, d1, d2, d3);

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
namespace infini {

void whereKernel(int dTypeIndex, void *inputX, void *inputY,
                 const uint8_t *condition, void *output, int a0, int a1, int a2,
                 int a3, int b0, int b1, int b2, int b3, int c0, int c1, int c2,
                 int c3, int d0, int d1, int d2, int d3) {
    int blocksize;
    int outputsize = d0 * d1 * d2 * d3;
    if (outputsize > 511) {
        blocksize = 1024;
    } else if (outputsize > 255) {
        blocksize = 512;
    } else if (outputsize > 127) {
        blocksize = 256;
    } else if (outputsize > 63) {
        blocksize = 128;
    } else if (outputsize > 31) {
        blocksize = 64;
    } else {
        blocksize = 32;
    }
    int gridsize = (outputsize + blocksize - 1) / blocksize;

    SWITCH_DTYPE(dTypeIndex)
}

} // namespace infini
