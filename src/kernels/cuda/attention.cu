#include "cuda/cuda_common.h"
const int Rq = 4;
const int Rv = 8; // 必须是4的倍数
const int Br = 16;
const int Bc = 16;
const int Bk = 4; // 必须是4的倍数

template <int Br, int Bc, int Rq>
__device__ void matmulRQK(const float *__restrict inputQ,
                          const float *__restrict inputK, float *shareQK,
                          float *shareVK, int N, int d, int width, int indQ,
                          int indK, float *val) {
    float a[4];
    for (int ph = 0; ph < width; ph++) {
        for (int index_k = 0; index_k < Bk; index_k++) {
            (float4 &)a[0] = (float4 &)
                inputK[(indK + index_k) * d + (threadIdx.y + ph * Bc) * Bk];
            for (int idk = 0; idk < Bk; idk++) {
                if (threadIdx.y < Bc) {
                    shareVK[(threadIdx.y * Bk + idk) * Bc * Bk +
                            threadIdx.x * Bk + index_k] = a[idk];
                    if (indK + index_k >= N ||
                        (threadIdx.y + ph * Bc) * Bk + idk >= d) {

                        shareVK[(threadIdx.y * Bk + idk) * Bc * Bk +
                                threadIdx.x * Bk + index_k] = 0.0f;
                    }
                }
            }
        }

        for (int index_q = 0; index_q < Rq; index_q++) {
            (float4 &)shareQK[(threadIdx.y * Rq + index_q) * Bc * Bk +
                              threadIdx.x * Bk] = (float4 &)
                inputQ[(indQ + index_q) * d + (threadIdx.x + ph * Bc) * Bk];
            for (int idk = 0; idk < Bk; idk++) {
                if (indQ + index_q >= N ||
                    (threadIdx.x + ph * Bc) * Bk + idk >= d) {
                    shareQK[(threadIdx.y * Rq + index_q) * Bc * Bk +
                            threadIdx.x * Bk + idk] = 0.0f;
                }
            }
        }
        __syncthreads();

        for (int index = 0; index < Bc * Bk; index++) {
            for (int index_q = 0; index_q < Rq; index_q++) {
                for (int index_k = 0; index_k < Bk; index_k++) {
                    val[index_q * Bk + index_k] = std::fma(
                        shareQK[(threadIdx.y * Rq + index_q) * Bc * Bk + index],
                        shareVK[index * Bc * Bk + threadIdx.x * Bk + index_k],
                        val[index_q * Bk + index_k]);
                }
            }
        }
        __syncthreads();
    }
}
template <int Br, int Bc, int Rq, int Rv>
__device__ void matmulSV(float *shareQK, const float *__restrict inputV,
                         float *shareVK, int N, int d, int j, int indQ,
                         int indK, int indV, float *val, float *newMax,
                         float *sumSV) {
    if (threadIdx.y < Bc) {
        for (int index_k = 0; index_k < Bk; index_k++) {
            for (int id = 0; id < (int)(Rv / 4); id++) {
                (float4 &)shareVK[(threadIdx.y * Bk + index_k) * Bc * Rv +
                                  threadIdx.x * Rv + id * 4] = (float4 &)
                    inputV[((threadIdx.y + j * Bc) * Bk + index_k) * d + indV +
                           id * 4];
            }
            for (int index_v = 0; index_v < Rv; index_v++) {
                if ((threadIdx.y + j * Bc) * Bk + index_k >= N ||
                    indV + index_v >= d) {
                    shareVK[(threadIdx.y * Bk + index_k) * Bc * Rv +
                            threadIdx.x * Rv + index_v] = 0.0f;
                }
            }
        }
    }
    for (int index_q = 0; index_q < Rq; index_q++) {
        for (int index_k = 0; index_k < Bk; index_k++) {
            if (indQ + index_q < N && indK + index_k < N) {
                shareQK[(threadIdx.y * Rq + index_q) * Bc * Bk +
                        threadIdx.x * Bk + index_k] =
                    __expf(val[index_q * Bk + index_k] - newMax[index_q]);
            } else {

                shareQK[(threadIdx.y * Rq + index_q) * Bc * Bk +
                        threadIdx.x * Bk + index_k] = 0.0f;
            }
        }
    }
    __syncthreads();

    for (int phc = 0; phc < Bc * Bk; phc++) {
        for (int index_q = 0; index_q < Rq; index_q++) {

            for (int index_v = 0; index_v < Rv; index_v++) {
                sumSV[index_q * Rv + index_v] +=
                    shareQK[(threadIdx.y * Rq + index_q) * Bc * Bk + phc] *
                    shareVK[phc * Bc * Rv + threadIdx.x * Rv + index_v];
            }
        }
    }
}
template <typename T> struct SumOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

template <typename T> struct MaxOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return max(a, b);
    }
};
template <template <typename> class ReductionOp, typename T,
          int thread_group_width = 32>
__inline__ __device__ T WarpAllReduce(T val) {
    for (int mask = thread_group_width / 2; mask > 0; mask >>= 1) {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }

    return val;
}

template <int Br, int Bc, int Rq, int Rv>
__global__ void _attentionKernel(const float *__restrict inputQ,
                                 const float *__restrict inputK,
                                 const float *__restrict inputV, int N, int d,
                                 float *__restrict output) {

    __shared__ float shareQK[Rq * Br * Bc * Bk];
    __shared__ float shareVK[Bk * Bc * Bc * Rv];

    float sumSV[Rq * Rv] = {0.0f};
    float newMax[Rq];
    float oldMax[Rq];
    float newSum[Rq] = {0.0f};

    float val[Rq * Bk];

    int indV = Rv * (threadIdx.x + blockIdx.x * blockDim.x);
    int indQ = Rq * (threadIdx.y + blockIdx.y * blockDim.y);

    for (int index_q = 0; index_q < Rq; index_q++) {
        newMax[index_q] = -__FLT_MAX__;
        oldMax[index_q] = -__FLT_MAX__;
    }

    int Tc = (N + Bc * Bk - 1) / (Bc * Bk);

    int width = (d + Bc * Bk - 1) / (Bc * Bk);
    for (int j = 0; j < Tc; j++) {

        int indK = Bk * (threadIdx.x + j * Bc);
        for (int index_q = 0; index_q < Rq; index_q++) {
            for (int index_k = 0; index_k < Bk; index_k++) {

                val[index_q * Bk + index_k] = 0.0f;
            }
        }
        matmulRQK<Br, Bc, Rq>(inputQ, inputK, shareQK, shareVK, N, d, width,
                              indQ, indK, val);
        for (int index_q = 0; index_q < Rq; index_q++) {
            float tmpReduceMax = -__FLT_MAX__;
            for (int index_k = 0; index_k < Bk; index_k++) {
                if (indQ + index_q < N && indK + index_k < N) {

                    tmpReduceMax =
                        max(tmpReduceMax, val[index_q * Bk + index_k]);
                }
            }
            __syncthreads();
            tmpReduceMax = WarpAllReduce<MaxOp, float, Bc>(tmpReduceMax);
            if (threadIdx.x == 0) {
                shareQK[threadIdx.y * Rq + index_q] = tmpReduceMax;
            }
            __syncthreads();
            float tmpReduceSum = 0.0f;
            for (int index_k = 0; index_k < Bk; index_k++) {
                if (indQ + index_q < N && indK + index_k < N) {
                    tmpReduceSum += __expf(val[index_q * Bk + index_k] -
                                           shareQK[threadIdx.y * Rq + index_q]);
                }
            }
            __syncthreads();
            tmpReduceSum = WarpAllReduce<SumOp, float, Bc>(tmpReduceSum);
            if (threadIdx.x == 0) {
                shareQK[threadIdx.y * Rq + index_q + Rq * Br] = tmpReduceSum;
            }
            __syncthreads();
            if (newMax[index_q] > shareQK[threadIdx.y * Rq + index_q]) {
                newSum[index_q] =
                    std::fma(shareQK[threadIdx.y * Rq + index_q + Rq * Br],
                             __expf(shareQK[threadIdx.y * Rq + index_q] -
                                    newMax[index_q]),
                             newSum[index_q]);
            } else {
                newSum[index_q] =
                    std::fma(newSum[index_q],
                             __expf(newMax[index_q] -
                                    shareQK[threadIdx.y * Rq + index_q]),
                             shareQK[threadIdx.y * Rq + index_q + Rq * Br]);

                newMax[index_q] = shareQK[threadIdx.y * Rq + index_q];
            }
            // PV
            for (int index_v = 0; index_v < Rv; index_v++) {
                sumSV[index_q * Rv + index_v] *=
                    __expf(oldMax[index_q] - newMax[index_q]);
            }
        }

        matmulSV<Br, Bc, Rq, Rv>(shareQK, inputV, shareVK, N, d, j, indQ, indK,
                                 indV, val, newMax, sumSV);

        for (int index_q = 0; index_q < Rq; index_q++) {
            oldMax[index_q] = newMax[index_q];
        }

        //__syncthreads();
    }
    for (int index_q = 0; index_q < Rq; index_q++) {
        float inv = __fdividef(1.0F, newSum[index_q]);
        for (int index_v = 0; index_v < Rv; index_v++) {
            sumSV[index_q * Rv + index_v] = sumSV[index_q * Rv + index_v] * inv;
        }
    }
    for (int index_q = 0; index_q < Rq; index_q++) {

        for (int id = 0; id < (int)(Rv / 4); id++) {
            if (indQ + index_q < N) {
                (float4 &)output[(indQ + index_q) * d + indV + id * 4] =
                    (float4 &)sumSV[index_q * Rv + id * 4];
            }
        }
    }
}
namespace infini {
void attentionKernel(const float *inputQ, const float *inputK,
                     const float *inputV, int N, int d, float *output) {
    int num_block_x = (d + Rv * Bc - 1) / (Rv * Bc);
    int num_block_y = (N + Rq * Br - 1) / (Rq * Br);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    dim3 block_dim(Bc, Br, 1);

    _attentionKernel<Br, Bc, Rq, Rv>
        <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
}
} // namespace infini
