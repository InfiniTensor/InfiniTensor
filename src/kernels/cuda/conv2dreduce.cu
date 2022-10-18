#include "cuda/cuda_common.h"

__global__ void conv2dreduce_kernel_(
    float *__restrict__ input,
    float *__restrict__ bias,
    float *__restrict__ output,
    const bool PReLU, const float paramReLU, const int n, const int f, const int h, const int w, 
    const int oh, const int ow, const int r, const int s, const int ph,
    const int pw, const int dh, const int dw, const int sh, const int sw)
{
    // output shape: (n, oh, ow, f)
    // input shape: (n, h, w, f, r, s)
    int nid = blockIdx.x, fid = blockIdx.y;
    int hid = threadIdx.x, wid = threadIdx.y;
    const int fchunck = r * s, wchunk = f * fchunck, hchunk = w * wchunk, nchunck = n * hchunk;
    float *nfinput = input + nid * nchunck + fid * fchunck;
    if (nid < n && fid < f && hid < oh && wid < ow) {
        float imm = 0.0;
        int ihst = hid * sh - ph;
        int iwst = wid * sw - pw;
        for (int ri = 0; ri < r; ++ri) {
            for (int si = 0; si < s; ++si) {
                int ihid = ihst + ri * dh;
                int iwid = iwst + si * dw;
                if (ihid >= 0 && ihid < h && iwid >= 0 && iwid < w) {
                    imm += *(nfinput + ihid * hchunk + iwid * wchunk + ri * s + si);
                }
            }
        }
        if (bias) {
            imm += bias[fid];
        }
        if (PReLU) {
            imm = imm > 0.0 ? imm : paramReLU * imm;
        }
        output[nid * (oh * ow * f) + hid * (ow * f) + wid * f + fid] = imm;
    }
}

__global__ void convTranspose2dreduce_kernel_(
    float *__restrict__ input,
    float *__restrict__ bias,
    float *__restrict__ output,
    const bool PReLU, const float paramReLU, const int n, const int f, const int h, const int w, 
    const int oh, const int ow, const int r, const int s, const int ph,
    const int pw, const int dh, const int dw, const int sh, const int sw)
{
    // assert dh = dw = 1
    int nid = blockIdx.x, fid = blockIdx.y;
    int hid = threadIdx.x, wid = threadIdx.y;
    const int fchunck = r * s, wchunk = f * fchunck, hchunk = w * wchunk, nchunck = n * hchunk;
    float *nfinput = input + nid * nchunck + fid * fchunck;
    // view as conv, the true ph and pw
    int tph = r - ph - 1, tpw = s - pw - 1;
    int th = (h - 1) * sh + 1, tw = (w - 1) * sw + 1;
    if (nid < n && fid < f && hid < oh && wid < ow) {
        float imm = 0.0;
        int ihst = hid - tph;
        int iwst = wid - tpw;
        for (int ri = 0; ri < r; ++ri) {
            for (int si = 0; si < s; ++si) {
                int ihid = ihst + r - ri - 1;
                int iwid = iwst + s - si - 1;
                if  (ihid >= 0 && ihid < th &&
                    iwid >= 0 && iwid < tw &&
                    (ihid % sh == 0) && (iwid % sw == 0))
                {
                    imm += *(nfinput + (ihid / sh) * hchunk + (iwid / sw) * wchunk + ri * s + si);
                }
            }
        }
        if (bias) {
            imm += bias[fid];
        }
        if (PReLU) {
            imm = imm > 0.0 ? imm : paramReLU * imm;
        }
        output[nid * (oh * ow * f) + hid * (ow * f) + wid * f + fid] = imm;
    }
}

namespace infini {

void conv2dreduce_kernel(float *input, float *bias, float *output, 
                          bool PReLU, float paramReLU, int n, int h, int w, int f,
                          int r, int s, int oh, int ow, int ph, int pw, int sh,
                          int sw, int dh, int dw)
{
    dim3 grid(n, f);
    dim3 block(oh, ow);
    cudaStream_t stream(cudaStreamPerThread);
    conv2dreduce_kernel_<<<grid, block, 0, stream>>>(input, bias, output, PReLU, paramReLU, n, f, h, w, 
                                        oh, ow, r, s, ph, pw, dh, dw, sh, sw);
}

void convTranspose2dreduce_kernel(float *input, float *bias, float *output, 
                          bool PReLU, float paramReLU, int n, int h, int w, int f,
                          int r, int s, int oh, int ow, int ph, int pw, int sh,
                          int sw, int dh, int dw)
{
    dim3 grid(n, f);
    dim3 block(oh, ow);
    cudaStream_t stream(cudaStreamPerThread);
    convTranspose2dreduce_kernel_<<<grid, block, 0,stream>>>(input, bias, output, PReLU, paramReLU, n, f, h, w, 
                                        oh, ow, r, s, ph, pw, dh, dw, sh, sw);
}
}
