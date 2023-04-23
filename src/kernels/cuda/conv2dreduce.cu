#include "cuda/cuda_common.h"
#include "nnet/dbg.h"

using dtype = float;

__global__ void conv2dreduce_kernel_(float *__restrict__ input,
                                     float *__restrict__ bias,
                                     float *__restrict__ output,
                                     const bool PReLU, const int n, const int f,
                                     const int h, const int w, const int oh,
                                     const int ow, const int r, const int s,
                                     const int ph, const int pw, const int dh,
                                     const int dw, const int sh, const int sw) {
    // output shape: (n, oh, ow, f)
    // input shape: (n, h, w, f, r, s)
    int nid = blockIdx.x, fid = blockIdx.y;
    int hid = threadIdx.x, wid = threadIdx.y;
    const int fchunck = r * s, wchunk = f * fchunck, hchunk = w * wchunk,
              nchunck = h * hchunk;
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
                    imm += *(nfinput + ihid * hchunk + iwid * wchunk + ri * s +
                             si);
                }
            }
        }
        if (bias) {
            imm += bias[fid];
        }
        if (PReLU) {
            imm = imm > 0.0 ? imm : 0.0;
        }
        output[nid * (oh * ow * f) + hid * (ow * f) + wid * f + fid] = imm;
    }
}
__global__ void convTranspose2dreduce_kernel2_(
    float *__restrict__ input, float *__restrict__ bias,
    float *__restrict__ output, const bool PReLU, const int n, const int f,
    const int h, const int w, const int oh, const int ow, const int r,
    const int s, const int ph, const int pw, const int dh, const int dw,
    const int sh, const int sw) {
    int warp_id = (blockDim.x / 32) * blockIdx.x + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int nid = warp_id / (f * oh * ow);
    int fid = (warp_id - nid * (f * oh * ow)) / (oh * ow);
    int hid = (warp_id - nid * (f * oh * ow) - fid * (oh * ow)) / ow;
    int wid = warp_id % ow;
    if (hid >= oh || wid >= ow || nid > n || fid > f)
        return;

    const int fchunck = r * s, wchunk = f * fchunck, hchunk = w * wchunk,
              nchunck = h * hchunk;
    float *nfinput = input + nid * nchunck + fid * fchunck;
    // view as conv, the true ph and pw
    int tph = r - ph - 1, tpw = s - pw - 1;
    int th = (h - 1) * sh + 1, tw = (w - 1) * sw + 1;

    float imm = 0.0;
    int ihst = hid - tph;
    int iwst = wid - tpw;
    for (int idx = lane; idx < r * s; idx += 32) {
        int ri = idx / s;
        int si = idx % s;
        int ihid = ihst + r - ri - 1;
        int iwid = iwst + s - si - 1;
        if (ihid >= 0 && ihid < th && iwid >= 0 && iwid < tw &&
            (ihid % sh == 0) && (iwid % sw == 0)) {
            imm += *(nfinput + (ihid / sh) * hchunk + (iwid / sw) * wchunk +
                     ri * s + si);
        }
    }

    for (int k = 16; k > 0; k >>= 1) {
        imm += __shfl_down_sync(0xffffffff, imm, k); // sum
    }
    if (lane == 0) {
        if (bias) {
            imm += bias[fid];
        }
        if (PReLU) {
            imm = imm > 0.0 ? imm : 0.0;
        }
        output[nid * (oh * ow * f) + hid * (ow * f) + wid * f + fid] = imm;
    }
}

__global__ void convTranspose2dreduce_kernel_(
    float *__restrict__ input, float *__restrict__ bias,
    float *__restrict__ output, const bool PReLU, const int n, const int f,
    const int h, const int w, const int oh, const int ow, const int r,
    const int s, const int ph, const int pw, const int dh, const int dw,
    const int sh, const int sw, const int block_x_num, const int block_y_num) {
    // assert dh = dw = 1
    int nid = blockIdx.x / block_x_num, fid = blockIdx.y / block_y_num;
    int hid = (blockIdx.x % block_x_num) * blockDim.x + threadIdx.x,
        wid = (blockIdx.y % block_y_num) * blockDim.y + threadIdx.y;
    if (hid >= oh || wid >= ow)
        return;
    const int fchunck = r * s, wchunk = f * fchunck, hchunk = w * wchunk,
              nchunck = h * hchunk;
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
                if (ihid >= 0 && ihid < th && iwid >= 0 && iwid < tw &&
                    (ihid % sh == 0) && (iwid % sw == 0)) {
                    imm += *(nfinput + (ihid / sh) * hchunk +
                             (iwid / sw) * wchunk + ri * s + si);
                }
            }
        }
        if (bias) {
            imm += bias[fid];
        }
        if (PReLU) {
            imm = imm > 0.0 ? imm : 0.0;
        }
        output[nid * (oh * ow * f) + hid * (ow * f) + wid * f + fid] = imm;
    }
}

// nhwrsc -> nhwc
__global__ void reduce_4x4(dtype *in, dtype *out, int act, const int N,
                           const int F, const int H, const int W, const int IH,
                           const int IW) {
// #define in_index(n, h, w, r, s, f)                                             \
//     ((((((n)*IH + h) * IW + w) * R + r) * S + s) * F + f)
#define in_index(n, h, w, f, r, s)                                             \
    ((((((n)*IH + h) * IW + w) * F + f) * R + r) * S + s)
#define out_index(n, h, w, f) (((((n)*H) + (h)) * W + (w)) * F + (f))
    const int R = 4, S = 4;
    const int n_tasks = N * F * H * W;
    int start = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = start; i < n_tasks; i += stride) {
        int t = i, n, f, h, w;
        f = t % F;
        t /= F;
        w = t % W;
        t /= W;
        h = t % H;
        t /= H;
        n = t;

        // unroll this 2-iter loop
        float sum = 0;
        int x, y;
        for (int r = (h + 1) & 1; r < R; r += 2) {
            x = (h + 1 - r) / 2;
            if (x >= 0 && x < IH) {
                for (int s = (w + 1) & 1; s < S; s += 2) {
                    y = (w + 1 - s) / 2;
                    if (y >= 0 && y < IW) {
                        sum += in[in_index(n, x, y, f, r, s)];
                        // if (i==0)
                        //     printf("TTT nhwf= %d,%d,%d,%d x=%d y=%d, v=%f,
                        //     index=%d, rsf %d %d %d\n", n, h, w,
                        //            f, x, y, in[in_index(n, x, y, r, s, f)],
                        //            in_index(n, x, y, r, s, f), r,s,f);
                    }
                }
            }
        }
        if (act == 0) {
            out[out_index(n, h, w, f)] = sum;
        } else if (act == 1) { // Relu
            out[out_index(n, h, w, f)] = sum > 0 ? sum : 0;
        } else if (act == 2) {
            out[out_index(n, h, w, f)] = tanhf(sum);
        }
    }
#undef in_index
#undef out_index
}

namespace infini {

void conv2dreduce_kernel(float *input, float *bias, float *output, bool PReLU,
                         int n, int h, int w, int f, int r, int s, int oh,
                         int ow, int ph, int pw, int sh, int sw, int dh,
                         int dw) {
    dim3 grid(n, f);
    dim3 block(oh, ow);
    // cudaStream_t stream(cudaStreamPerThread);
    conv2dreduce_kernel_<<<grid, block, 0>>>(input, bias, output, PReLU, n, f,
                                             h, w, oh, ow, r, s, ph, pw, dh, dw,
                                             sh, sw);
}

void convTranspose2dreduce_kernel(float *input, float *bias, float *output,
                                  int act, int n, int h, int w, int f, int r,
                                  int s, int oh, int ow, int ph, int pw, int sh,
                                  int sw, int dh, int dw) {
    dim3 grid(n, f);
    dim3 block(oh, ow);
    // cudaStream_t stream(cudaStreamPerThread);
    // puts("convTranspose2dreduce_kernel is executed");
    if (r == 4 && s == 4 && sh == 2 && sw == 2) {
        const int M = r * s * f, N = n * h * w;
        reduce_4x4<<<(M * N + 127) / 128, 128>>>(input, output, act, n, f, oh,
                                                 ow, h, w);
    } else {
        // puts("why use this conv2dreduce");
        // block.x = 32;
        // block.y = 32;
        // int block_x_num = (oh + block.x - 1) / block.x;
        // int block_y_num = (ow + block.y - 1) / block.y;
        // grid.x = n * (block_x_num);
        // grid.y = f * (block_y_num);
        // convTranspose2dreduce_kernel_<<<grid, block, 0>>>(
        //     input, bias, output, (bool)act, n, f, h, w, oh, ow, r, s, ph, pw,
        //     dh, dw, sh, sw, block_x_num, block_y_num);

        block.x = 128;
        block.y = 1;
        grid.x = (n * f * ow * oh + block.x / 32 - 1) / (block.x / 32);
        grid.y = 1;
        convTranspose2dreduce_kernel2_<<<grid, block, 0>>>(
            input, bias, output, (bool)act, n, f, h, w, oh, ow, r, s, ph, pw,
            dh, dw, sh, sw);
    }
}
} // namespace infini
