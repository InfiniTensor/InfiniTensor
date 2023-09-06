#include "cuda/cuda_common.h"
#include "utils/small_array.h"
#include <cub/cub.cuh>

struct __align__(8) MD { // update the global max and sum, store the output at
                         // max_tmp and sum_tmp
    float max_tmp;       // store max
    float sum_tmp;       // store sum
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
template <int BLOCK_DIM>
__launch_bounds__(BLOCK_DIM) __global__
    void _softmax_kernel(float *input, float *output, int size,
                         infini::SmallArray inputShape, int axis, int nDims,
                         int stride) { // if set axis = 1, inputShape=[I,J,K,S]
    int tid = 0;                       // tid = i(JKS) + j(KS) + k(S) + s
    int dimsize = inputShape.data[axis]; // set axis = 1, dimsize = J
    // blockDim.x = size/dimsize = IKS
    // blockIdx.x = i(KS) + k(S) + s,blockIdx.x%stride = k(S) + s

    tid = blockIdx.x % stride + (blockIdx.x - blockIdx.x % stride) *
                                    dimsize; // now, tid = i(JKS) + k(S) + s;

    MD md_partial;
    md_partial.max_tmp = -__FLT_MAX__;
    md_partial.sum_tmp = 0.0f;
    for (int id = threadIdx.x; id < dimsize; id += blockDim.x) {
        MD md_input;
        md_input.max_tmp = input[tid + id * stride];
        md_input.sum_tmp = 1.0f;
        md_partial = reduce_md_op(md_partial,
                                  md_input); // reduce the data to one block
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
    //-----------------
    float max_total, sum_inverse_total;
    max_total = md_total.max_tmp;
    sum_inverse_total = __fdividef(1.0F, md_total.sum_tmp);
    for (int id = threadIdx.x; id < dimsize; id += blockDim.x) {
        output[tid + id * stride] =
            __expf(input[tid + id * stride] - max_total) * sum_inverse_total;
    }
}
namespace infini {
void softmax_kernel(float *input, float *output, int size,
                    SmallArray inputShape, int axis, int nDims, int stride) {
    int dimsize = inputShape.data[axis];
    int num_blocks = size / dimsize;
    if (dimsize > 1023) {
        int BLOCK_DIM = 1024;
        _softmax_kernel<1024><<<num_blocks, BLOCK_DIM>>>(
            input, output, size, inputShape, axis, nDims, stride);
    } else if (dimsize > 511) {
        int BLOCK_DIM = 512;
        _softmax_kernel<512><<<num_blocks, BLOCK_DIM>>>(
            input, output, size, inputShape, axis, nDims, stride);
    } else if (dimsize > 255) {
        int BLOCK_DIM = 256;
        _softmax_kernel<256><<<num_blocks, BLOCK_DIM>>>(
            input, output, size, inputShape, axis, nDims, stride);
    } else if (dimsize > 127) {
        int BLOCK_DIM = 128;
        _softmax_kernel<128><<<num_blocks, BLOCK_DIM>>>(
            input, output, size, inputShape, axis, nDims, stride);
    } else if (dimsize > 63) {
        int BLOCK_DIM = 64;
        _softmax_kernel<64><<<num_blocks, BLOCK_DIM>>>(
            input, output, size, inputShape, axis, nDims, stride);
    } else {
        int BLOCK_DIM = 32;
        _softmax_kernel<32><<<num_blocks, BLOCK_DIM>>>(
            input, output, size, inputShape, axis, nDims, stride);
    }
}
} // namespace infini
