#include "cuda/cuda_common.h"
#include "utils/small_array.h"
#include <cub/cub.cuh>

#define BLOCK_DIM_x 32
#define BLOCK_DIM_y 32

struct __align__(8) MD {
    float max_tmp; // store max
    float sum_tmp; // store sum
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

__global__ void _softmax_kernel(float *input, float *output, int size,
                                infini::SmallArray inputShape, int axis,
                                int nDims, float *res_sum, float *res_max) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // i < inputShape[axis]
    int j = threadIdx.y + blockIdx.y * blockDim.y; // j < size/inputShape[axis]
    int size_x = inputShape.data[axis];
    int size_y = size / inputShape.data[axis];
    if (j < size_y && i < size_x) {
        int v = j;
        int tid = 0;
        int stride = 1;
        int temp = 1;
        int ijks = 0;
        for (int k = nDims - 1; k >= 0; --k) {
            if (k == 0) {
                ijks = v; // i
            } else if (k == axis) {
                v /= 1;
            } else {
                ijks = v % inputShape.data[k]; // s,k,j
                v /= inputShape.data[k];
            }
            if (k == axis) {
                stride = temp;
            } else {
                tid += ijks * temp;
            }
            temp *= inputShape.data[k];
        }
        MD md_partial;
        md_partial.max_tmp = -__FLT_MAX__;
        md_partial.sum_tmp = 0.0f;
        for (int id = threadIdx.x; id < size_x; id += blockDim.x) {
            MD md_input;
            md_input.max_tmp = input[tid + id * stride];
            md_input.sum_tmp = 1.0f;
            md_partial = reduce_md_op(md_partial,
                                      md_input); // reduce the data to one block
        }
        typedef cub::BlockReduce<MD, BLOCK_DIM_x> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;

        MD md_block =
            BlockReduce(temp_storage).Reduce(md_partial, reduce_md_op);
        if (threadIdx.x ==
            0) { // must set threadIdx.x = 0 write the output to memory
            res_sum[j] = md_block.sum_tmp;
            res_max[j] = md_block.max_tmp;
        }
        __syncthreads();
        //-----------------

        output[tid + i * stride] =
            __expf(input[tid + i * stride] - res_max[j]) *
            __fdividef(1.0F, res_sum[j]);
    }
}
namespace infini {
void softmax_kernel(float *input, float *output, int size,
                    SmallArray inputShape, int axis, int nDims) {
    int size_x = inputShape.data[axis];
    int size_y = size / inputShape.data[axis];
    int num_block_x = ceil(size_x / (double)BLOCK_DIM_x);
    int num_block_y = ceil(size_y / (double)BLOCK_DIM_y);
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    float *res_sum, *res_max;
    cudaMalloc((void **)&res_sum, size_y * sizeof(float));
    cudaMalloc((void **)&res_max, size_y * sizeof(float));
    _softmax_kernel<<<grid_dim, block_dim>>>(input, output, size, inputShape,
                                             axis, nDims, res_sum, res_max);
}
} // namespace infini
