#include "cuda/cuda_common.h"

__global__ void bias2prelu_kernel_(
    float *__restrict__ input,
    float *__restrict__ bias,
    float *__restrict__ output,
    const bool PReLU, const float paramReLU,
    const int n, const int h, const int w, const int c)
{
    int nid = blockIdx.x, hid = blockIdx.y;
    int wid = threadIdx.x, cid = threadIdx.y;

    int offset = nid * h * w * c + hid * w * c + wid * c + cid;
    float imm = bias[cid] + input[offset];
    if (PReLU) {
        imm = (imm > 0.0) ? imm : paramReLU * paramReLU;
    }
    output[offset] = imm;
}

namespace infini {

void bias2prelu_kernel(float *input, float *bias, float *output,
                      int n, int h, int w, int c, bool PReLU, float paramPReLU)
{
    dim3 grid(n, h);
    dim3 block(w, c);
    cudaStream_t stream(cudaStreamPerThread);
    bias2prelu_kernel_<<<grid, block, 0, stream>>>(input, bias, output, 
                                                    PReLU, paramPReLU, n, h, w, c);

}

}