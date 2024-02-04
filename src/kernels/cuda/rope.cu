#include "core/common.h"
#include "cuda/cuda_common.h"
#include "cuda/cuda_utility.h"
#include "utils/small_array.h"

constexpr unsigned int num_threads() { return 32 * 4; }
constexpr int thread_work_size() { return 4; }
constexpr int block_work_size() { return thread_work_size() * num_threads(); }

// gridDim (batch, seq_len, dim_model / 1024),   blockDim (1024, 1, 1)
template <class T>
__global__ void _rope_kernel(int* pos, void *in, void *out, int size, int dim_model, int dim_head, int hidden_stride, int pos_stride) {
    int batch_id = blockIdx.x;
    int target_pos = pos[batch_id * pos_stride + blockIdx.y];
    int ith = blockIdx.z * blockDim.x + threadIdx.x;
    int col = ith % dim_head;
    int offset = batch_id * hidden_stride + blockIdx.y * dim_model;

    if (ith >= dim_model)
        return;
    int half_dim = dim_head / 2;
    if (col < half_dim) {
        float freq = target_pos * powf(10000, -float(col * 2) / dim_head);
        float cos_freq = cos(freq);
        float sin_freq = sin(freq);
        ((T *)out)[offset + ith] =
            ((T *)in)[offset + ith] * T(cos_freq) - ((T *)in)[offset + ith + half_dim] * T(sin_freq);
    } else {
        float freq = target_pos * powf(10000, -float((col - half_dim) * 2) / dim_head);
        float cos_freq = cos(freq);
        float sin_freq = sin(freq);
        ((T *)out)[offset + ith] =
            ((T *)in)[offset + ith] * T(cos_freq) + ((T *)in)[offset + ith - half_dim] * T(sin_freq);
    }
}


#define CASE(T)                                                                \
    _rope_kernel<DT_CUDA<T>::t><<<gridsize, blocksize>>>(                 \
        pos, input, output, size, dim_model, dim_head, hidden_stride, pos_stride);

#define SWITCH_DTYPE(DTYPE)                                                    \
    switch (DTYPE) {                                                           \
    case 1:                                                                    \
        CASE(1)                                                                \
        break;                                                                 \
    case 2:                                                                    \
        CASE(2)                                                                \
        break;                                                                 \
    case 3:                                                                    \
        CASE(3)                                                                \
        break;                                                                 \
    case 4:                                                                    \
        CASE(4)                                                                \
        break;                                                                 \
    case 5:                                                                    \
        CASE(5)                                                                \
        break;                                                                 \
    case 6:                                                                    \
        CASE(6)                                                                \
        break;                                                                 \
    case 7:                                                                    \
        CASE(7)                                                                \
        break;                                                                 \
    case 10:                                                                   \
        CASE(10)                                                               \
        break;                                                                 \
    case 11:                                                                   \
        CASE(11)                                                               \
        break;                                                                 \
    case 12:                                                                   \
        CASE(12)                                                               \
        break;                                                                 \
    case 13:                                                                   \
        CASE(13)                                                               \
        break;                                                                 \
    case 16:                                                                   \
        CASE(16)                                                               \
        break;                                                                 \
    default:                                                                   \
        IT_TODO_HALT();                                                        \
    }

namespace infini {
void rope_kernel(int dType, int * pos, void *input, void *output, int size, int dim_model, int dim_head, int hidden_stride, int pos_stride) {
    dim3 blocksize = dim3(1024,1,1);
    dim3 gridsize = dim3(1, 1, 4);
    SWITCH_DTYPE(dType)
}

} // namespace infini
