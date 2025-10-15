#include "cuda/cuda_common.h"
#include <cmath>
#include <cuda_fp16.h>

// utility: convert half to float
__device__ inline float half2f(const __half &h) { return __half2float(h); }
__device__ inline __half f2half(float f) { return __float2half(f); }

// ------------------ float kernel ------------------
// Each block handles one (n, c) pair. threads per block = up to 1024 (chosen).
template <typename T, typename AccT>
__global__ void
instance_norm_kernel(const T *__restrict__ input,
                     const T *__restrict__ scale, // length C
                     const T *__restrict__ bias,  // can be nullptr
                     T *__restrict__ output, int N, int C, int inner_size,
                     float eps) {
    // block index corresponds to (n*c + c_idx)
    int nc = blockIdx.x;
    int n = nc / C;
    int c = nc % C;

    int tid = threadIdx.x;
    int threads = blockDim.x;
    int total = inner_size;

    // base pointer for this (n,c)
    const T *base = input + (size_t)nc * (size_t)inner_size;
    T *out_base = output + (size_t)nc * (size_t)inner_size;

    // 1) compute mean (in AccT)
    AccT sum = 0;
    for (int i = tid; i < total; i += threads) {
        AccT v = static_cast<AccT>(base[i]);
        sum += v;
    }
    // block-wide reduction (shared mem)
    __shared__ AccT
        s_sum[1024]; // safe because threads<=1024; template may set lower
    s_sum[tid] = sum;
    __syncthreads();

    // tree-reduce in shared memory
    for (int offset = threads >> 1; offset > 0; offset >>= 1) {
        if (tid < offset)
            s_sum[tid] += s_sum[tid + offset];
        __syncthreads();
    }
    AccT mean = static_cast<AccT>(0);
    if (tid == 0) {
        mean = s_sum[0] / static_cast<AccT>(total);
        // write mean to s_sum[0] for second-pass use (reuse)
        s_sum[0] = mean;
    }
    __syncthreads();
    mean = s_sum[0];

    // 2) compute variance (E[(x-mean)^2])
    AccT sqsum = 0;
    for (int i = tid; i < total; i += threads) {
        AccT v = static_cast<AccT>(base[i]);
        AccT diff = v - mean;
        sqsum += diff * diff;
    }
    s_sum[tid] = sqsum;
    __syncthreads();
    for (int offset = threads >> 1; offset > 0; offset >>= 1) {
        if (tid < offset)
            s_sum[tid] += s_sum[tid + offset];
        __syncthreads();
    }
    AccT var = static_cast<AccT>(0);
    if (tid == 0) {
        var = s_sum[0] / static_cast<AccT>(total);
        s_sum[0] = var; // store var
    }
    __syncthreads();
    var = s_sum[0];

    // 3) normalize and write output
    AccT denom = rsqrtf(var + static_cast<AccT>(eps)); // 1/sqrt(var+eps)
    // load scale and bias for channel c
    AccT s = static_cast<AccT>(scale[c]);
    AccT b = bias ? static_cast<AccT>(bias[c]) : static_cast<AccT>(0);

    for (int i = tid; i < total; i += threads) {
        AccT v = static_cast<AccT>(base[i]);
        AccT normed = (v - mean) * denom * s + b;
        out_base[i] = static_cast<T>(normed);
    }
}

namespace infini {
void InstanceNormKernel(const float *input, const float *scale,
                        const float *bias, float *output, int N, int C,
                        int inner_size, float eps) {
    int outer = N * C;
    if (outer == 0)
        return;
    // choose threads: min(1024, nearest power-of-two <= inner_size or 256
    // default)
    int threads = 256;
    if (inner_size >= 512)
        threads = 512;
    if (inner_size >= 1024)
        threads = 1024;
    if (threads > 1024)
        threads = 1024;

    dim3 blocks(outer);
    dim3 tthreads(threads);
    // instantiate template: T=float, AccT=float
    instance_norm_kernel<float, float>
        <<<blocks, tthreads, 0, CUDAStream::getCurrentStream()>>>(
            input, scale, bias, output, N, C, inner_size, eps);
}

void InstanceNormKernel(const __half *input, const __half *scale,
                        const __half *bias, __half *output, int N, int C,
                        int inner_size, float eps) {
    int outer = N * C;
    if (outer == 0)
        return;
    int threads = 256;
    if (inner_size >= 512)
        threads = 512;
    if (inner_size >= 1024)
        threads = 1024;
    if (threads > 1024)
        threads = 1024;

    dim3 blocks(outer);
    dim3 tthreads(threads);
    // instantiate template: T=__half, AccT=float
    instance_norm_kernel<__half, float>
        <<<blocks, tthreads, 0, CUDAStream::getCurrentStream()>>>(
            input, scale, bias, output, N, C, inner_size, eps);
}
} // namespace infini