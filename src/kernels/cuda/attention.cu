#include "cuda/cuda_common.h"

template <int Br, int Bc>
__device__ float matmul(const float *__restrict A, const float *__restrict B,
                        int d, int indA, int indB) {
    float sum_qk = 0.0f;
    for (int index = 0; index < d; index++) {
        sum_qk += A[indA * d + index] * B[indB * d + index];
    }
    return sum_qk;
}
template <int Br, int Bc>
__device__ float matmulShare(const float *__restrict inputQ,
                             const float *__restrict inputK, float *Qds,
                             float *Kds, int N, int d, int width, int indQ,
                             int indK) {
    float sum_qk = 0.0f;
    for (int ph = 0; ph < width; ph++) {
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
    return sum_qk;
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
          int thread_group_width = warpSize>
__inline__ __device__ T WarpAllReduce(T val) {
    for (int mask = thread_group_width / 2; mask > 0; mask >>= 1) {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }

    return val;
}
template <int Br, int Bc>
__global__ void _attentionKernel(const float *__restrict inputQ,
                                 const float *__restrict inputK,
                                 const float *__restrict inputV, int N, int d,
                                 float *__restrict output) {

    int Tc = (N + Bc - 1) / Bc;

    __shared__ float sumQK[Br * Bc];
    float sumSV;
    __shared__ float block_max[Br];
    __shared__ float block_sum[Br];
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
    newSum = 0.0f;

    float out = 0.0f;
    for (int j = 0; j < Tc; j++) {
        sumSV = 0.0f;
        int indK = threadIdx.x + j * Bc;
        float sum_qk = 0.0f;
        float tmp_qk = 0.0f;
        sum_qk = matmulShare<Br, Bc>(inputQ, inputK, Qds, Kds, N, d, gridDim.x,
                                     indQ, indK);
        if (indQ < N && indK < N) {
            tmp_qk = sum_qk;

        } else {
            sum_qk = -__FLT_MAX__;
            tmp_qk = 0.0f;
        }
        __syncthreads();
        // softmax reduce
        sum_qk = WarpAllReduce<MaxOp, float, Bc>(sum_qk);
        if (threadIdx.x == 0) {
            block_max[threadIdx.y] = sum_qk;
        }
        __syncthreads();
        float localMax = block_max[threadIdx.y];
        //--------------------
        float sum_s = 0.0f;
        if (indQ < N && indK < N) {
            sum_s = __expf(tmp_qk - localMax);
        }
        sum_s = WarpAllReduce<SumOp, float, Bc>(sum_s);
        if (threadIdx.x == 0) {
            block_sum[threadIdx.y] = sum_s;
        }
        __syncthreads();
        float localSum = block_sum[threadIdx.y];
        if (newMax > localMax) {
            newSum = std::fma(localSum, __expf(localMax - newMax), newSum);
            // newSum = newSum + localSum * __expf(localMax - newMax);
        } else {
            newSum = std::fma(newSum, __expf(newMax - localMax), localSum);
            // newSum = localSum + newSum * __expf(newMax - localMax);
            newMax = localMax;
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
            sumQK[threadIdx.y * Bc + threadIdx.x] = __expf(tmp_qk - newMax);
        } else {
            sumQK[threadIdx.y * Bc + threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int phc = 0; phc < Bc; phc++) {
            sumSV = std::fma(sumQK[threadIdx.y * Bc + phc],
                             Vds[threadIdx.x * Bc + phc], sumSV);
        }
        out = std::fma(__expf(oldMax - newMax), out, sumSV);
        // out = __expf(oldMax - newMax) * out + sumSV;
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
