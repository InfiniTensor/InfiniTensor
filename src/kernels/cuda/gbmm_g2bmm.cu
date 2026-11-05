#include "cuda/gbmm_g2bmm.cuh"
#include "cuda/gbmm_g2bmm.h"
#include "cuda/cuda_common.h"

__global__ void g2bmm_half_kernel(const half *q, const half *k, half *y,
                                  int bs, int n, int m, int w, int d) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(bs) * n * (2 * w + 1);
    if (idx >= total)
        return;
    int window = idx % (2 * w + 1);
    int row = (idx / (2 * w + 1)) % n;
    int batch = idx / (static_cast<size_t>(n) * (2 * w + 1));
    int neighbor = row + (window - w) * d;
    float sum = 0.f;
    if (neighbor >= 0 && neighbor < n) {
        size_t qBase = (static_cast<size_t>(batch) * n + row) * m;
        size_t kBase = (static_cast<size_t>(batch) * n + neighbor) * m;
        for (int channel = 0; channel < m; ++channel)
            sum += __half2float(q[qBase + channel]) *
                   __half2float(k[kBase + channel]);
    }
    y[idx] = __float2half(sum);
}

__global__ void gbmm_half_kernel(const half *weights, const half *values,
                                 half *y, int bs, int n, int m, int w, int d) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(bs) * n * m;
    if (idx >= total)
        return;
    int channel = idx % m;
    int row = (idx / m) % n;
    int batch = idx / (static_cast<size_t>(n) * m);
    size_t weightBase = (static_cast<size_t>(batch) * n + row) * (2 * w + 1);
    float sum = 0.f;
    for (int window = 0; window < 2 * w + 1; ++window) {
        int neighbor = row + (window - w) * d;
        if (neighbor >= 0 && neighbor < n) {
            size_t valueIndex = (static_cast<size_t>(batch) * n + neighbor) * m +
                                channel;
            sum += __half2float(weights[weightBase + window]) *
                   __half2float(values[valueIndex]);
        }
    }
    y[idx] = __float2half(sum);
}

namespace infini {

void _sg2bmm(float *__restrict__ q, float *__restrict__ k,
             float *__restrict__ y, int bs, int n, int m, int w, int d) {
    sg2bmm(q, k, y, bs, n, m, w, d);
}

void _sgbmml(float *__restrict__ q, float *__restrict__ k,
             float *__restrict__ y, int bs, int n, int m, int w, int d) {
    sgbmml(q, k, y, bs, n, m, w, d);
}

void _sg2bmm(const half *q, const half *k, half *y, int bs, int n, int m,
              int w, int d) {
    size_t total = static_cast<size_t>(bs) * n * (2 * w + 1);
    g2bmm_half_kernel<<<(total + 255) / 256, 256, 0,
                        CUDAStream::getCurrentStream()>>>(q, k, y, bs, n, m,
                                                          w, d);
}

void _sgbmml(const half *q, const half *k, half *y, int bs, int n, int m,
              int w, int d) {
    size_t total = static_cast<size_t>(bs) * n * m;
    gbmm_half_kernel<<<(total + 255) / 256, 256, 0,
                       CUDAStream::getCurrentStream()>>>(q, k, y, bs, n, m,
                                                         w, d);
}

} // namespace infini
