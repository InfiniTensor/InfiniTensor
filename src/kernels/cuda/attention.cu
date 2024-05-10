#include "cuda/cuda_common.h"
template <int Br, int Bc>
__global__ void _attentionKernel(const float *__restrict inputQ,
                                 const float *__restrict inputK,
                                 const float *__restrict inputV, int N, int d,

                                 float *__restrict output) {

    int Tc = (N + Bc - 1) / Bc;

    __shared__ float sumQK[Br * Bc];
    __shared__ float sumSV[Br * Bc];
    __shared__ float block_max[Br * Bc];
    __shared__ float block_sum[Br * Bc];
    __shared__ float Vds[Bc * Bc];
    __shared__ float Qds[Br * Bc];
    __shared__ float Kds[Bc * Bc];
    int indV = threadIdx.x + blockIdx.x * blockDim.x;
    int indQ = threadIdx.y + blockIdx.y * blockDim.y;
    float newMax;
    float oldMax;
    float newSum;
    newMax = -__FLT_MAX__;
    oldMax = -__FLT_MAX__;
    newSum = 1.0f;

    float out = 0.0f;
    for (int j = 0; j < Tc; j++) {
        sumSV[threadIdx.y * Bc + threadIdx.x] = 0.0f;
        int indK = threadIdx.x + j * Bc;
        float sum_qk = 0.0f;
        for (int ph = 0; ph < gridDim.x; ph++) {
            if (indQ < N && threadIdx.x + ph * Bc < d) {
                Qds[threadIdx.y * Bc + threadIdx.x] =
                    inputQ[indQ * d + threadIdx.x + ph * Bc];
            } else {
                Qds[threadIdx.y * Bc + threadIdx.x] = 0.0f;
            }
            if (threadIdx.y < Bc) {
                Kds[threadIdx.y * Bc + threadIdx.x] = 0.0f;
            }
            if (threadIdx.y < Bc) {
                if (indK < N && threadIdx.y + ph * Bc < d) {
                    Kds[threadIdx.y * Bc + threadIdx.x] =
                        inputK[indK * d + threadIdx.y + ph * Bc];
                }
            }

            __syncthreads();
            for (int index = 0; index < Bc; index++) {
                sum_qk = std::fma(Qds[threadIdx.y * Bc + index],
                                  Kds[index * Bc + threadIdx.x], sum_qk);
            }
            __syncthreads();
        }

        if (indQ < N && indK < N) {
            block_max[threadIdx.y * Bc + threadIdx.x] = sum_qk;
            block_sum[threadIdx.y * Bc + threadIdx.x] = 1.0f;
            sumQK[threadIdx.y * Bc + threadIdx.x] = sum_qk;
        } else {
            block_max[threadIdx.y * Bc + threadIdx.x] = -__FLT_MAX__;
            block_sum[threadIdx.y * Bc + threadIdx.x] = 0.0f;
            sumQK[threadIdx.y * Bc + threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int strip = Bc / 2; strip > 0; strip /= 2) {
            if (threadIdx.x < strip) {
                if (block_max[threadIdx.y * Bc + threadIdx.x] >
                    block_max[threadIdx.y * Bc + threadIdx.x + strip]) {
                    block_sum[threadIdx.y * Bc + threadIdx.x] =
                        block_sum[threadIdx.y * Bc + threadIdx.x] +
                        block_sum[threadIdx.y * Bc + threadIdx.x + strip] *
                            __expf(block_max[threadIdx.y * Bc + threadIdx.x +
                                             strip] -
                                   block_max[threadIdx.y * Bc + threadIdx.x]);
                } else {
                    block_sum[threadIdx.y * Bc + threadIdx.x] =
                        block_sum[threadIdx.y * Bc + threadIdx.x + strip] +
                        block_sum[threadIdx.y * Bc + threadIdx.x] *
                            __expf(block_max[threadIdx.y * Bc + threadIdx.x] -
                                   block_max[threadIdx.y * Bc + threadIdx.x +
                                             strip]);
                    block_max[threadIdx.y * Bc + threadIdx.x] =
                        block_max[threadIdx.y * Bc + threadIdx.x + strip];
                }
            }
            __syncthreads();
        }

        if (newMax > block_max[threadIdx.y * Bc]) {
            newSum = newSum + block_sum[threadIdx.y * Bc] *
                                  __expf(block_max[threadIdx.y * Bc] - newMax);
        } else {
            newSum = block_sum[threadIdx.y * Bc] +
                     newSum * __expf(newMax - block_max[threadIdx.y * Bc]);
            newMax = block_max[threadIdx.y * Bc];
        }
        if (threadIdx.y < Bc) {
            if (threadIdx.y + j * Bc < N && indV < d) {
                Vds[threadIdx.x * Bc + threadIdx.y] =
                    inputV[(threadIdx.y + j * Bc) * d + indV];
            } else {
                Vds[threadIdx.x * Bc + threadIdx.y] = 0.0f;
            }
        }
        if (indQ < N && indK < N) {
            sumQK[threadIdx.y * Bc + threadIdx.x] =
                __expf(sumQK[threadIdx.y * Bc + threadIdx.x] - newMax);
        } else {
            sumQK[threadIdx.y * Bc + threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int phc = 0; phc < Bc; phc++) {
            sumSV[threadIdx.y * Bc + threadIdx.x] = std::fma(
                sumQK[threadIdx.y * Bc + phc], Vds[threadIdx.x * Bc + phc],
                sumSV[threadIdx.y * Bc + threadIdx.x]);
        }
        out = __expf(oldMax - newMax) * out +
              sumSV[threadIdx.y * Bc + threadIdx.x];
        oldMax = newMax;

        //__syncthreads();
    }
    if (indQ < N && indV < d) {
        output[indQ * d + indV] = out * __fdividef(1.0F, newSum);
    }
}
namespace infini {
void attentionKernel(const float *inputQ, const float *inputK,
                     const float *inputV, int N, int d, float *output) {
    int Br = 32;
    int Bc = 32; // Br>=Bc

    int num_block_x = (d + Bc - 1) / Bc;
    int num_block_y = (N + Br - 1) / Br;
    dim3 block_dim(Bc, Br, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);

    _attentionKernel<32, 32>
        <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
}
} // namespace infini
