#include "cuda/cuda_transpose.h"
#include <stdio.h>
#include <math.h>

constexpr unsigned int num_threads() { return 32 * 4; }
constexpr int thread_work_size() { return 4; }
constexpr int block_work_size() { return thread_work_size() * num_threads(); }


__global__ void _transpose_kernel(float *a, float *c, int dim_0, int dim_1, int dim_2, int dim_3,
                                          int p_0, int p_1, int p_2, int p_3) {

    int src_dim[4] = {dim_0, dim_1, dim_2, dim_3};
    int stride_dim[4] = {dim_1*dim_2*dim_3, dim_2*dim_3, dim_3, 1};
    int permute[4] = {p_0, p_1, p_2, p_3};
    int dst_dim[4] = {src_dim[p_0], src_dim[p_1], src_dim[p_2], src_dim[p_3]};
    int n = dim_0 * dim_1 * dim_2 * dim_3;

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        int c0_index = i / (dst_dim[1] * dst_dim[2] * dst_dim[3]);
        int c1_index = (i %  (dst_dim[1] * dst_dim[2] * dst_dim[3])) / (dst_dim[2] * dst_dim[3]);
        int c2_index = ((i % (dst_dim[1] * dst_dim[2] * dst_dim[3])) % (dst_dim[2] * dst_dim[3])) / dst_dim[3];
        int c3_index = ((i % (dst_dim[1] * dst_dim[2] * dst_dim[3])) % (dst_dim[2] * dst_dim[3])) % dst_dim[3]; 
        int new_0 = c0_index * stride_dim[permute[0]];
        int new_1 = c1_index * stride_dim[permute[1]];
        int new_2 = c2_index * stride_dim[permute[2]];
        int new_3 = c3_index * stride_dim[permute[3]];
        int src_address = new_0 + new_1 + new_2 + new_3; 
        c[i] = a[src_address];
    }
}

namespace infini {
void transpose_kernel(float *a, float *c, int dim_0, int dim_1, int dim_2, int dim_3,
                                          int p_0, int p_1, int p_2, int p_3) {
    int blocksize = block_work_size();
    int gridsize = (dim_0*dim_1*dim_2*dim_3 + block_work_size() - 1) / block_work_size();
    _transpose_kernel<<<blocksize, gridsize>>>(a,c,dim_0,dim_1,dim_2,dim_3,p_0,p_1,p_2,p_3);
}

}; // namespace infini
