#include "cuda/cuda_common.h"
#include <cub/cub.cuh>
struct __align__(8) MD {
    float max_tmp;
    float sum_tmp;
};
__device__ __forceinline__ MD reduce_md_op(MD a, MD b) {
    bool a_bigger = (a.max_tmp > b.max_tmp);
    MD bigger = a_bigger ? a : b;
    MD smaller = a_bigger ? b : a;
    MD res;
    res.sum_tmp = bigger.sum_tmp +
                  smaller.sum_tmp * __expf(smaller.max_tmp - bigger.max_tmp);
    res.max_tmp = bigger.max_tmp;
    return res;
}

#define max_function(a, b) ((a) > (b) ? (a) : (b))

template <int BLOCK_DIM>
__launch_bounds__(BLOCK_DIM) __global__
    void _attentionKernel(const float *inputQ, const float *inputK,
                          const float *inputV, float *inputS, int N, int d,
                          float *output) {
    MD md_partial;
    md_partial.max_tmp = -__FLT_MAX__;
    md_partial.sum_tmp = 0.0f;
    int phNumN = (N + BLOCK_DIM - 1) / BLOCK_DIM;
    int i = blockIdx.x; // i must < N
    for (int phn = 0; phn < phNumN; phn++) {

        int j = threadIdx.x + phn * BLOCK_DIM;
        MD md_input;
        if (j < N) {
            float sum_s = 0.0f;
            for (int index = 0; index < d; index++) {
                sum_s += inputQ[i * d + index] * inputK[j * d + index];
            }
            inputS[i * N + j] = sum_s;
            // printf("S--%d:%.4e\n",i * N + j,inputS[i * N + j]);
            md_input.max_tmp = sum_s;
            md_input.sum_tmp = 1.0f;
        } else {
            md_input.max_tmp = -__FLT_MAX__;
            md_input.sum_tmp = 0.0f;
        }
        md_partial = reduce_md_op(md_partial, md_input);
    }
    typedef cub::BlockReduce<MD, BLOCK_DIM> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    __shared__ MD md_total;
    MD md_block = BlockReduce(temp_storage).Reduce(md_partial, reduce_md_op);
    if (threadIdx.x ==
        0) { // must set threadIdx.x = 0 write the output to memory
        md_total = md_block;
    }
    __syncthreads();
    // printf("max:%.4e\n",md_total.max_tmp);
    for (int phn = 0; threadIdx.x + phn * BLOCK_DIM < N; phn++) {
        int j = threadIdx.x + phn * BLOCK_DIM;
        inputS[i * N + j] = __expf(inputS[i * N + j] - md_total.max_tmp) *
                            __fdividef(1.0F, md_total.sum_tmp);
        // printf("S:%.4e\n",inputS[i * N + j]);
    }
    __syncthreads();

    for (int phd = 0; threadIdx.x + phd * BLOCK_DIM < d; phd++) {
        int j = threadIdx.x + phd * BLOCK_DIM;
        float sum_o = 0;
        for (int index = 0; index < N; index++) {
            sum_o += inputS[i * N + index] * inputV[index * d + j];
        }
        output[i * d + j] = sum_o;
    }
}
namespace infini {
void attentionKernel(const float *inputQ, const float *inputK,
                     const float *inputV, int N, int d, float *output) {

    float *inputS;
    cudaMalloc((void **)&inputS, N * N * sizeof(float));
    int nd = max_function(N, d);
    if (nd > 1023) {
        _attentionKernel<1024>
            <<<N, 1024>>>(inputQ, inputK, inputV, inputS, N, d, output);
    } else if (nd > 511) {
        _attentionKernel<512>
            <<<N, 512>>>(inputQ, inputK, inputV, inputS, N, d, output);
    } else if (nd > 255) {
        _attentionKernel<256>
            <<<N, 256>>>(inputQ, inputK, inputV, inputS, N, d, output);
    } else if (nd > 63) {
        _attentionKernel<64>
            <<<N, 64>>>(inputQ, inputK, inputV, inputS, N, d, output);
    } else if (nd > 15) {
        _attentionKernel<16>
            <<<N, 16>>>(inputQ, inputK, inputV, inputS, N, d, output);
    } else {
        _attentionKernel<8>
            <<<N, 8>>>(inputQ, inputK, inputV, inputS, N, d, output);
    }
}
} // namespace infini