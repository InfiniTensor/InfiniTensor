#include "cuda/cuda_common.h"
#include "cuda/softmax.h"
#include <cub/cub.cuh>

struct __align__(8) MD {
    float data;
    float d;
};

__device__ __forceinline__ MD reduce_md_op(MD a, MD b) {
    bool a_bigger = (a.data > b.data);
    MD bigger_m = a_bigger ? a : b;
    MD smaller_m = a_bigger ? b : a;
    MD res;
    res.d = bigger_m.d + smaller_m.d * __expf(smaller_m.data - bigger_m.data);
    res.data = bigger_m.data;
    return res;
}

template <int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void online_softmax(const float *__restrict in, float *__restrict out,
                        int dimSize, int stride) {

    // reposition in and out to data for the current vector
    int blockOffset = blockIdx.x;
    if (blockIdx.x >= stride) {
        int tmp = blockIdx.x % stride;
        blockOffset = tmp + (blockIdx.x - tmp) * dimSize;
    }
    in += blockOffset;
    out += blockOffset;

    MD md_partial;
    md_partial.data = -FLT_MAX;
    md_partial.d = 0.0F;

    for (int elem_id = threadIdx.x; elem_id < dimSize;
         elem_id += THREADBLOCK_SIZE) {
        MD new_elem;
        new_elem.data = in[elem_id * stride];
        new_elem.d = 1.0F;
        md_partial = reduce_md_op(md_partial, new_elem);
    }

    // blockreduce for THREADBLOCK_SIZE threads.
    // The actrual threads num used in the block is "dimsSize"
    typedef cub::BlockReduce<MD, THREADBLOCK_SIZE> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ MD md_total;

    MD md = BlockReduce(temp_storage).Reduce(md_partial, reduce_md_op);
    if (threadIdx.x == 0)
        md_total = md;
    __syncthreads();

    float d_total_inverse = __fdividef(1.0F, md_total.d);
    for (int elem_id = threadIdx.x; elem_id < dimSize;
         elem_id += THREADBLOCK_SIZE)
        out[elem_id * stride] =
            __expf(in[elem_id * stride] - md_total.data) * d_total_inverse;
}

namespace infini {
void softmax_kernel(int max_threadblock_size, int blockNum, float *in,
                    float *out, int dimSize, int stride) {
    if (max_threadblock_size >= 255)
        online_softmax<256><<<blockNum, 256>>>(in, out, dimSize, stride);
    else if (max_threadblock_size >= 128)
        online_softmax<128><<<blockNum, 128>>>(in, out, dimSize, stride);
    else if (max_threadblock_size >= 64)
        online_softmax<64><<<blockNum, 64>>>(in, out, dimSize, stride);
    else
        online_softmax<32><<<blockNum, 32>>>(in, out, dimSize, stride);
}
} // namespace infini
