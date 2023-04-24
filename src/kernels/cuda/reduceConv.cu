#include "core/common.h"
#include <vector>
using namespace std;

template <class T>
__global__ void reduce_merge_conv_3x3_1x1(
    T *__restrict__ input, T *__restrict__ output, T *__restrict__ bias,
    const int N, const int H, const int W, const int F, const int N_offset,
    const int H_offset, const int W_offset, const int F_offset,
    const int out_N_offset, const int out_F_offset, const int out_H_offset,
    const int out_W_offset, const int num) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num) {
        int tmptid = tid;
        const int n = (tmptid / out_N_offset);
        tmptid -= n * out_N_offset;
        const int f = tmptid / out_F_offset;
        tmptid -= f * out_F_offset;
        const int h = tmptid / out_H_offset;
        tmptid -= h * out_H_offset;
        const int w = tmptid / out_W_offset;

        const int noff = n * N_offset;
        const int hoff = h * H_offset;
        const int woff = w * W_offset;
        const int foff = f * F_offset;
        input += noff + foff + woff + hoff;
        T res = 0;
        res += input[4];
        res += input[9];

        if (h < H - 1) {
            res += input[H_offset + 7];
            if (w < W - 1)
                res += input[H_offset + W_offset + 8];
            if (w > 0)
                res += input[H_offset - W_offset + 6];
        }
        if (h > 0) {
            res += input[1 - H_offset];
            if (w < W - 1)
                res += input[W_offset - H_offset + 2];
            if (w > 0)
                res += input[-1 * H_offset - W_offset];
        }
        if (w < W - 1)
            res += input[5 + W_offset];
        if (w > 0)
            res += input[3 - W_offset];
        output[tid] = max(res + bias[f], 0.f);
    }
}

template <class T>
__global__ void reduce_merge_conv_3x3(
    T *__restrict__ input, T *__restrict__ output, T *__restrict__ bias,
    const int N, const int H, const int W, const int F, const int N_offset,
    const int H_offset, const int W_offset, const int F_offset,
    const int out_N_offset, const int out_F_offset, const int out_H_offset,
    const int out_W_offset, const int num, const int act) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num) {
        int tmptid = tid;
        const int n = (tmptid / out_N_offset);
        tmptid -= n * out_N_offset;
        const int f = tmptid / out_F_offset;
        tmptid -= f * out_F_offset;
        const int h = tmptid / out_H_offset;
        tmptid -= h * out_H_offset;
        const int w = tmptid / out_W_offset;

        const int noff = n * N_offset;
        const int hoff = h * H_offset;
        const int woff = w * W_offset;
        const int foff = f * F_offset;
        input += noff + foff + woff + hoff;
        T res = 0;
        res += input[4];

        if (h < H - 1) {
            res += input[H_offset + 7];
            if (w < W - 1)
                res += input[H_offset + W_offset + 8];
            if (w > 0)
                res += input[H_offset - W_offset + 6];
        }
        if (h > 0) {
            res += input[1 - H_offset];
            if (w < W - 1)
                res += input[W_offset - H_offset + 2];
            if (w > 0)
                res += input[-1 * H_offset - W_offset];
        }
        if (w < W - 1)
            res += input[5 + W_offset];
        if (w > 0)
            res += input[3 - W_offset];
        if (act) {
            // output[tid] = max(res + bias[f], 0.f);
            // HACK: temperaly remove bias
            output[tid] = max(res, 0.f);
        } else {
            // output[tid] = res + bias[f];
            // HACK: temperaly remove bias
            output[tid] = res;
        }
    }
}

template <class T>
__global__ void
reduce_2(T *__restrict__ input, T *__restrict__ output, T *__restrict__ bias,
         const int N, const int F, const int H, const int W, const int N_offset,
         const int F_offset, const int H_offset, const int W_offset,
         const int out_N_offset, const int out_F_offset, const int out_H_offset,
         const int out_W_offset, const int num) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num) {
        int tmptid = tid;
        const int n = tmptid / out_N_offset;
        tmptid -= n * out_N_offset;
        const int f = tmptid / out_F_offset;
        tmptid -= f * out_F_offset;
        const int h = tmptid / out_H_offset;
        tmptid -= h * out_H_offset;
        const int w = tmptid / out_W_offset;

        const int noff = n * N_offset;
        const int foff = f * F_offset * 4;
        const int hoff = h * H_offset;
        const int woff = w * W_offset;
        input += noff + foff + woff + hoff;
        T res = input[0];
        if (w != W - 1)
            res += input[F_offset * 2 + 3];
        if (h != H - 1) {
            res += input[F_offset + 3 * H_offset];
            if (w != W - 1)
                res += input[F_offset * 3 + 3 * H_offset + 3];
        }
        // output[tid] = max(res + bias[f], 0.f);
        // HACK: temperaly remove bias
        output[tid] = max(res, 0.f);
    }
}

__global__ void reduceConvRxSToNCHWKernel(
    float *__restrict__ input, float *__restrict__ bias,
    float *__restrict__ output, const int act, const int n, const int f,
    const int h, const int w, const int oh, const int ow, const int r,
    const int s, const int ph, const int pw, const int dh, const int dw) {
    // input shape: (n, h, w, f, r, s)
    // output shape: (n, f, oh, ow)
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_N_offset = f * oh * ow, out_F_offset = oh * ow,
              out_H_offset = ow, out_W_offset = 1;
    const int num = out_N_offset * n;
    if (tid < num) {
        // output index
        int tmptid = tid;
        const int nid = (tmptid / out_N_offset);
        tmptid -= nid * out_N_offset;
        const int fid = tmptid / out_F_offset;
        tmptid -= fid * out_F_offset;
        const int hid = tmptid / out_H_offset;
        tmptid -= hid * out_H_offset;
        const int wid = tmptid / out_W_offset;

        // Input index
        const int fchunck = r * s, wchunk = f * fchunck, hchunk = w * wchunk,
                  nchunck = h * hchunk;
        float *__restrict__ nfinput = input + nid * nchunck + fid * fchunck;
        float imm = 0.0;
        const int ihst = hid, iwst = wid;
        for (int ri = 0; ri < r; ++ri) {
            for (int si = 0; si < s; ++si) {
                int ihid = ihst + (ri - r / 2) * dh;
                int iwid = iwst + (si - s / 2) * dw;
                if (ihid >= 0 && ihid < h && iwid >= 0 && iwid < w) {
                    imm += *(nfinput + ihid * hchunk + iwid * wchunk + ri * s +
                             si);
                }
            }
        }
        if (bias) {
            imm += bias[fid];
        }
        if (act) {
            imm = imm > 0.0 ? imm : 0;
        }
        output[tid] = imm;
    }
}

namespace infini {

void hetConvToMMReduce(int n, int h, int w, int f, float *input, float *output,
                       float *bias) {
    const int kBlockSize = 128;
    vector<int> in_params = {n, h, w, f}; // NHWF
    vector<int> out_params = {n, f, h, w};
    int in_base = 10;
    int out_base = 1;
    vector<int> in_offsets;
    vector<int> out_offsets;
    for (int i = 0; i < 4; ++i) {
        in_offsets.push_back(in_base);
        in_base *= in_params[3 - i];
        out_offsets.push_back(out_base);
        out_base *= out_params[3 - i];
    }
    reduce_merge_conv_3x3_1x1<float>
        <<<(out_base + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
            input, output, bias, in_params[0], in_params[1], in_params[2],
            in_params[3], in_offsets[3], in_offsets[2], in_offsets[1],
            in_offsets[0], out_offsets[3], out_offsets[2], out_offsets[1],
            out_offsets[0], out_base);
}

void conv5x5ToConv3x3Reduce(int n, int f, int h, int w, float *input,
                            float *output, float *bias) {
    const int kBlockSize = 128;
    vector<int> params{n, f, h, w}; // NFHW
    vector<int> ranges(4);
    ranges[3] = params[3] + 2;
    ranges[2] = params[2] + 2;
    ranges[1] = params[1] * 4;
    ranges[0] = params[0];

    int in_base = 1;
    int out_base = 1;
    vector<int> in_offsets;
    vector<int> out_offsets;
    for (int i = 0; i < 4; ++i) {
        in_offsets.push_back(in_base);
        in_base *= ranges[3 - i];
        out_offsets.push_back(out_base);
        out_base *= params[3 - i];
    }
    reduce_2<float><<<(out_base + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
        input, output, bias, params[0], params[1], params[2], params[3],
        in_offsets[3], in_offsets[2], in_offsets[1], in_offsets[0],
        out_offsets[3], out_offsets[2], out_offsets[1], out_offsets[0],
        out_base);
}

// [NHW,FRS] -> [NFHW]
void conv3x3ToReduce(int n, int h, int w, int f, float *input, float *output,
                     float *bias) {
    const int kBlockSize = 128;
    vector<int> in_params = {n, h, w, f}; // NHWF
    vector<int> out_params = {n, f, h, w};
    int in_base = 9;
    int out_base = 1;
    vector<int> in_offsets;
    vector<int> out_offsets;
    for (int i = 0; i < 4; ++i) {
        in_offsets.push_back(in_base);
        in_base *= in_params[3 - i];
        out_offsets.push_back(out_base);
        out_base *= out_params[3 - i];
    }
    reduce_merge_conv_3x3<float>
        <<<(out_base + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
            input, output, bias, in_params[0], in_params[1], in_params[2],
            in_params[3], in_offsets[3], in_offsets[2], in_offsets[1],
            in_offsets[0], out_offsets[3], out_offsets[2], out_offsets[1],
            out_offsets[0], out_base, 0);
}

void reduceConvRxSToNCHW(float *input, float *bias, float *output, int act,
                         int n, int h, int w, int f, int r, int s, int oh,
                         int ow, int ph, int pw, int sh, int sw, int dh,
                         int dw) {
    IT_ASSERT(sh == 1 && sw == 1,
              "reduceConvRxSToNCHWKernel_kernel only support sh=sw=1");
    IT_ASSERT(dh == 1 && dw == 1,
              "reduceConvRxSToNCHWKernel_kernel only support dh=dw=1");
    const int blocksize = 512;
    const int gridsize = (n * f * oh * ow + blocksize - 1) / blocksize;

    cudaStream_t stream(cudaStreamPerThread);
    reduceConvRxSToNCHWKernel<<<gridsize, blocksize, 0, stream>>>(
        input, bias, output, act, n, f, h, w, oh, ow, r, s, ph, pw, dh, dw);
}

} // namespace infini
