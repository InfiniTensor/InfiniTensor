#include "core/common.h"
#include "cuda/cuda_common.h"
#include "cuda/cuda_utility.h"
#include "utils/small_array.h"

template<class T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(uint32_t(-1), val, mask);
  return val;
}

/* Calculate the sum of all elements in a block */
template<class T>
__inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if (lane == 0)
    shared[wid] = val;

  __syncthreads();

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
  val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);
  return val;
}

template <class T>
__global__ void _rmsnorm_kernel(void *in, void *weight, void *out, int num_tokens, int hidden_size) {
    __shared__ float s_variance;
    float variance = 0.0f;

    for(int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x){
        const float x = ((float*) in)[blockIdx.x * hidden_size + idx];
        variance += x * x; 
    }
    variance = blockReduceSum<float>(variance);
    if(threadIdx.x == 0){
        s_variance = rsqrtf(variance / hidden_size + 0.00001f);
    }
    __syncthreads();

    for(int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x){
        float x = ((float*) in)[blockIdx.x * hidden_size + idx];
        ((T*)out)[blockIdx.x * hidden_size + idx] = ((T)(x * s_variance)) * ((T*)weight)[idx];
    }
}


#define CASE(T)                                                                \
    _rmsnorm_kernel<DT_CUDA<T>::t>                                             \
        <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>           \
        (input, weight, output, num_tokens, hidden_size);

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
void rmsnorm_kernel(int dType, void *input, void *weight, void *output, 
                    int num_tokens, int hidden_size) {
    dim3 blocksize = dim3(std::min(hidden_size, 1024));
    dim3 gridsize = dim3(num_tokens);
    SWITCH_DTYPE(dType)
}

} // namespace infini
