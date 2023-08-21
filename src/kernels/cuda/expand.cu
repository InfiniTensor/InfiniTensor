
#include "cuda/cuda_common.h"
// outputsize默认是inputsize的倍数

__global__ void _expand_kernel(float *d_input, float *d_output, int shape,
                               int inputsize) { // d_input,d_output都是1D向量

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int outputsize = shape * inputsize;
    if (i < outputsize) {
        d_output[i] = d_input[i % inputsize];
    }
}

namespace infini {
void expand_kernel(float *d_input, float *d_output, int shape, int inputsize) {
    int outputsize = shape * inputsize;
    int blocksize = 32 * 16;
    int gridsize = (outputsize + blocksize - 1) / blocksize;
    _expand_kernel<<<blocksize, gridsize>>>(d_input, d_output, shape,
                                            inputsize);
}
} // namespace infini
