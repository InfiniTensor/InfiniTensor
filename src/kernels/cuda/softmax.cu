#include "cuda/cuda_common.h"
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <cfloat>
template <typename T> struct __align__(8) DataMaxSum {
    float max_tmp;
    float sum_tmp;
};

template <typename T> struct SoftmaxScalar;

template <> struct SoftmaxScalar<float> {
    __device__ __forceinline__ static float exp(float x) { return __expf(x); }
    __device__ __forceinline__ static float neg_inf() { return -FLT_MAX; }
    __device__ __forceinline__ static float zero() { return 0.0f; }
    __device__ __forceinline__ static float one() { return 1.0f; }
};

template <> struct SoftmaxScalar<half> {
    __device__ __forceinline__ static half exp(half x) { return hexp(x); }
    __device__ __forceinline__ static half neg_inf() {
        return __float2half(-FLT_MAX);
    }
    __device__ __forceinline__ static half zero() {
        return __float2half(0.0f);
    }
    __device__ __forceinline__ static half one() {
        return __float2half(1.0f);
    }
};

template <typename T>
__device__ __forceinline__ T softmax_exp(T x) {
    return SoftmaxScalar<T>::exp(x);
}

template <typename T>
__device__ __forceinline__ T softmax_neg_inf() {
    return SoftmaxScalar<T>::neg_inf();
}

template <typename T>
__device__ __forceinline__ T softmax_zero() {
    return SoftmaxScalar<T>::zero();
}

template <typename T>
__device__ __forceinline__ T softmax_one() {
    return SoftmaxScalar<T>::one();
}

template <typename T>
__device__ __forceinline__ DataMaxSum<T> reduce_dms_op(DataMaxSum<T> a,
                                                       DataMaxSum<T> b) {
    bool a_bigger = (a.max_tmp > b.max_tmp);
    DataMaxSum<T> bigger = a_bigger ? a : b;
    DataMaxSum<T> smaller = a_bigger ? b : a;
    bigger.sum_tmp = bigger.sum_tmp +
                     smaller.sum_tmp *
                         softmax_exp<float>(smaller.max_tmp - bigger.max_tmp);

    return bigger;
}
template <typename T, int BLOCK_DIM>
__launch_bounds__(BLOCK_DIM) __global__ void _blockSoftmaxKernel(
    T *__restrict input, T *__restrict output, int size, int dimsize,
    int stride) { // if set axis = 1, inputShape=[I,J,K,S]
                  // tid = i(JKS) + j(KS) + k(S) + s

    // blockDim.x = size/dimsize = IKS
    // blockIdx.x = i(KS) + k(S) + s,blockIdx.x%stride = k(S) + s

    int tid =
        blockIdx.x % stride + (blockIdx.x - blockIdx.x % stride) *
                                  dimsize; // now, tid = i(JKS) + k(S) + s;

    DataMaxSum<T> dms_partial;
    dms_partial.max_tmp = softmax_neg_inf<float>();
    dms_partial.sum_tmp = softmax_zero<float>();
    DataMaxSum<T> dms_input;
    int remain = dimsize % BLOCK_DIM;
    int step = (dimsize - remain) / BLOCK_DIM + 1; // step <= numPerThread

    if (threadIdx.x < remain) {
        for (int ind = 0; ind < step; ind++) {
            dms_input.max_tmp =
                input[tid + (threadIdx.x * step + ind) * stride];

            dms_input.sum_tmp = softmax_one<float>();
            dms_partial =
                reduce_dms_op<T>(dms_partial,
                                 dms_input); // reduce the data to one block
        }
    } else {
        for (int ind = 0; ind < step - 1; ind++) {
            dms_input.max_tmp =
                input[tid + (remain * step +
                             (threadIdx.x - remain) * (step - 1) + ind) *
                                stride];

            dms_input.sum_tmp = softmax_one<float>();
            dms_partial =
                reduce_dms_op<T>(dms_partial,
                                 dms_input); // reduce the data to one block
        }
    }

    typedef cub::BlockReduce<DataMaxSum<T>, BLOCK_DIM> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ DataMaxSum<T> dms_total;
    DataMaxSum<T> dms_block =
        BlockReduce(temp_storage).Reduce(dms_partial, reduce_dms_op<T>);
    if (threadIdx.x ==
        0) { // must set threadIdx.x = 0 write the output to memory
        dms_total = dms_block;
    }
    __syncthreads();
    //-----------------
    if (threadIdx.x < remain) {
        for (int ind = 0; ind < step; ind++) {

            output[tid + (threadIdx.x * step + ind) * stride] =
                softmax_exp<float>(
                    static_cast<float>(input[tid +
                        (threadIdx.x * step + ind) * stride]) -
                    dms_total.max_tmp) /
                dms_total.sum_tmp;
        }
    } else {
        for (int ind = 0; ind < step - 1; ind++) {

            output[tid +
                   (remain * step + (threadIdx.x - remain) * (step - 1) + ind) *
                       stride] =
                softmax_exp<float>(
                    static_cast<float>(input[tid +
                          (remain * step +
                           (threadIdx.x - remain) * (step - 1) + ind) *
                              stride]) -
                    dms_total.max_tmp) /
                dms_total.sum_tmp;
        }
    }
}

template <typename T, int BLOCK_DIM, int numPerThread>
__global__ void
_blockSoftmaxKernel(T *__restrict input, T *__restrict output, int size,
                    int dimsize,
                    int stride) { // if set axis = 1, inputShape=[I,J,K,S]
                                  // tid = i(JKS) + j(KS) + k(S) + s

    // blockDim.x = size/dimsize = IKS
    // blockIdx.x = i(KS) + k(S) + s,blockIdx.x%stride = k(S) + s

    int tid =
        blockIdx.x % stride + (blockIdx.x - blockIdx.x % stride) *
                                  dimsize; // now, tid = i(JKS) + k(S) + s;
    int remain = dimsize % BLOCK_DIM;
    int step = (dimsize - remain) / BLOCK_DIM + 1; // step <= numPerThread
    float dataPerThread[numPerThread];

    DataMaxSum<T> dms_partial;
    dms_partial.max_tmp = softmax_neg_inf<float>();
    dms_partial.sum_tmp = softmax_zero<float>();
    DataMaxSum<T> dms_input;
    if (threadIdx.x < remain) {
        for (int ind = 0; ind < step; ind++) {
            dataPerThread[ind] =
                input[tid + (threadIdx.x * step + ind) * stride];
            dms_input.max_tmp = dataPerThread[ind];
            dms_input.sum_tmp = softmax_one<float>();
            dms_partial =
                reduce_dms_op<T>(dms_partial,
                                 dms_input); // reduce the data to one block
        }
    } else {
        for (int ind = 0; ind < step - 1; ind++) {
            dataPerThread[ind] =
                input[tid + (remain * step +
                             (threadIdx.x - remain) * (step - 1) + ind) *
                                stride];
            dms_input.max_tmp = dataPerThread[ind];
            dms_input.sum_tmp = softmax_one<float>();
            dms_partial =
                reduce_dms_op<T>(dms_partial,
                                 dms_input); // reduce the data to one block
        }
    }

    typedef cub::BlockReduce<DataMaxSum<T>, BLOCK_DIM> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ DataMaxSum<T> dms_total;
    DataMaxSum<T> dms_block =
        BlockReduce(temp_storage).Reduce(dms_partial, reduce_dms_op<T>);
    if (threadIdx.x ==
        0) { // must set threadIdx.x = 0 write the output to memory
        dms_total = dms_block;
    }
    __syncthreads();
    //-----------------
    if (threadIdx.x < remain) {
        for (int ind = 0; ind < step; ind++) {
            output[tid + (threadIdx.x * step + ind) * stride] =
                softmax_exp<float>(dataPerThread[ind] - dms_total.max_tmp) /
                dms_total.sum_tmp;
        }
    } else {
        for (int ind = 0; ind < step - 1; ind++) {
            output[tid +
                   (remain * step + (threadIdx.x - remain) * (step - 1) + ind) *
                       stride] =
                softmax_exp<float>(dataPerThread[ind] - dms_total.max_tmp) /
                dms_total.sum_tmp;
        }
    }
}

template <typename T> struct SumOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

template <typename T> struct MaxOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a > b ? a : b;
    }
};
template <template <typename> class ReductionOp, typename T,
          int thread_group_width>
__inline__ __device__ T WarpAllReduce(T val) {
    for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template <template <typename> class ReductionOp, int thread_group_width>
__inline__ __device__ half WarpAllReduce(half val) {
    for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
        unsigned short bits = __half_as_ushort(val);
        unsigned int other_bits =
            __shfl_xor_sync(0xffffffff, static_cast<unsigned int>(bits), mask);
        unsigned short other_bits_short = static_cast<unsigned short>(other_bits);
        val = ReductionOp<half>()(val, __ushort_as_half(other_bits_short));
    }
    return val;
}

template <typename T, int BLOCK_DIM_x, int BLOCK_DIM_y, int numPerThreadx>
__global__ void _warpSoftmaxKernel(T *__restrict input, T *__restrict output,
                                   int size, int dimsize, int stride) {
    int otherIdx = blockIdx.x * blockDim.y + threadIdx.y;
    int otherSize = size / dimsize;
    int tid = otherIdx % stride + (otherIdx - otherIdx % stride) * dimsize;
    float dataPerThreadx[numPerThreadx];
    if (otherIdx < otherSize) {

        __shared__ float max_total[BLOCK_DIM_y];
        __shared__ float sum_total[BLOCK_DIM_y];
        float max_data = softmax_neg_inf<float>();

        for (int ph = 0; threadIdx.x + ph * BLOCK_DIM_x < dimsize; ph++) {
            dataPerThreadx[ph] =
                input[tid + (threadIdx.x + ph * BLOCK_DIM_x) * stride];
            max_data = max_data > dataPerThreadx[ph] ? max_data
                                                     : dataPerThreadx[ph];
        }

        max_data = WarpAllReduce<MaxOp, float, BLOCK_DIM_x>(max_data);

        if (threadIdx.x == 0)
            max_total[threadIdx.y] = max_data;
        //--------------------------------------------
        float sum_data = softmax_zero<float>();

        for (int ph = 0; threadIdx.x + ph * BLOCK_DIM_x < dimsize; ph++) {
            dataPerThreadx[ph] =
                softmax_exp<float>(dataPerThreadx[ph] - max_total[threadIdx.y]);
            sum_data += dataPerThreadx[ph];
        }

        sum_data = WarpAllReduce<SumOp, float, BLOCK_DIM_x>(sum_data);

        if (threadIdx.x == 0)
            sum_total[threadIdx.y] = sum_data;
        //--------------------------------------------

        for (int ph = 0; threadIdx.x + ph * BLOCK_DIM_x < dimsize; ph++) {
            output[tid + (threadIdx.x + ph * BLOCK_DIM_x) * stride] =
                dataPerThreadx[ph] / sum_total[threadIdx.y];
        }
    }
}
//-----------------

//-----------------
namespace infini {
void softmax_kernel(int num_blocks, float *input, float *output, int size,
                    int dimsize, int stride) {

    if (dimsize > 1024 * 128) {

        int BLOCK_DIM = 1024;
        _blockSoftmaxKernel<float, 1024>
            <<<num_blocks, BLOCK_DIM, 0, CUDAStream::getCurrentStream()>>>
            (input, output, size, dimsize, stride);
    } else if (dimsize > 1024 * 64) {

        int BLOCK_DIM = 1024;
        _blockSoftmaxKernel<float, 1024, 128>
            <<<num_blocks, BLOCK_DIM, 0, CUDAStream::getCurrentStream()>>>
            (input, output, size, dimsize, stride);
    } else if (dimsize > 1024 * 32) {

        int BLOCK_DIM = 1024;
        _blockSoftmaxKernel<float, 1024, 64>
            <<<num_blocks, BLOCK_DIM, 0, CUDAStream::getCurrentStream()>>>
            (input, output, size, dimsize, stride);
    } else if (dimsize > 1024 * 16) {

        int BLOCK_DIM = 1024;
        _blockSoftmaxKernel<float, 1024, 32>
            <<<num_blocks, BLOCK_DIM, 0, CUDAStream::getCurrentStream()>>>
            (input, output, size, dimsize, stride);
    } else if (dimsize > 1024 * 4) {

        int BLOCK_DIM = 1024;
        _blockSoftmaxKernel<float, 1024, 16>
            <<<num_blocks, BLOCK_DIM, 0, CUDAStream::getCurrentStream()>>>
            (input, output, size, dimsize, stride);
    } else if (dimsize > 1024) {

        int BLOCK_DIM = 1024;
        _blockSoftmaxKernel<float, 1024, 4>
            <<<num_blocks, BLOCK_DIM, 0, CUDAStream::getCurrentStream()>>>
            (input, output, size, dimsize, stride);
    } else if (dimsize > 31) {
        int BLOCK_DIM_x = 32;
        int BLOCK_DIM_y = 32;
        int num_block_x = (num_blocks + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        _warpSoftmaxKernel<float, 32, 32, 32>
            <<<grid_dim, block_dim, 0, CUDAStream::getCurrentStream()>>>
            (input, output, size, dimsize, stride);
    } else if (dimsize > 15) {
        int BLOCK_DIM_x = 16;
        int BLOCK_DIM_y = 64;
        int num_block_x = (num_blocks + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        _warpSoftmaxKernel<float, 16, 64, 2>
            <<<grid_dim, block_dim, 0, CUDAStream::getCurrentStream()>>>
            (input, output, size, dimsize, stride);
    } else if (dimsize > 7) {
        int BLOCK_DIM_x = 8;
        int BLOCK_DIM_y = 128;
        int num_block_x = (num_blocks + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        _warpSoftmaxKernel<float, 8, 128, 2>
            <<<grid_dim, block_dim, 0, CUDAStream::getCurrentStream()>>>
            (input, output, size, dimsize, stride);
    } else {
        int BLOCK_DIM_x = 4;
        int BLOCK_DIM_y = 256;
        int num_block_x = (num_blocks + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        _warpSoftmaxKernel<float, 4, 256, 2>
            <<<grid_dim, block_dim, 0, CUDAStream::getCurrentStream()>>>
            (input, output, size, dimsize, stride);
    }
}
//------------------
void softmax_kernel(int num_blocks, half *input, half *output, int size,
                    int dimsize, int stride) {

    if (dimsize > 1024 * 128) {

        int BLOCK_DIM = 1024;
        _blockSoftmaxKernel<half, 1024>
            <<<num_blocks, BLOCK_DIM, 0, CUDAStream::getCurrentStream()>>>
            (input, output, size, dimsize, stride);
    } else if (dimsize > 1024 * 64) {

        int BLOCK_DIM = 1024;
        _blockSoftmaxKernel<half, 1024, 128>
            <<<num_blocks, BLOCK_DIM, 0, CUDAStream::getCurrentStream()>>>
            (input, output, size, dimsize, stride);
    } else if (dimsize > 1024 * 32) {

        int BLOCK_DIM = 1024;
        _blockSoftmaxKernel<half, 1024, 64>
            <<<num_blocks, BLOCK_DIM, 0, CUDAStream::getCurrentStream()>>>
            (input, output, size, dimsize, stride);
    } else if (dimsize > 1024 * 16) {

        int BLOCK_DIM = 1024;
        _blockSoftmaxKernel<half, 1024, 32>
            <<<num_blocks, BLOCK_DIM, 0, CUDAStream::getCurrentStream()>>>
            (input, output, size, dimsize, stride);
    } else if (dimsize > 1024 * 4) {

        int BLOCK_DIM = 1024;
        _blockSoftmaxKernel<half, 1024, 16>
            <<<num_blocks, BLOCK_DIM, 0, CUDAStream::getCurrentStream()>>>
            (input, output, size, dimsize, stride);
    } else if (dimsize > 1024) {

        int BLOCK_DIM = 1024;
        _blockSoftmaxKernel<half, 1024, 4>
            <<<num_blocks, BLOCK_DIM, 0, CUDAStream::getCurrentStream()>>>
            (input, output, size, dimsize, stride);
    } else if (dimsize > 31) {
        int BLOCK_DIM_x = 32;
        int BLOCK_DIM_y = 32;
        int num_block_x = (num_blocks + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        _warpSoftmaxKernel<half, 32, 32, 32>
            <<<grid_dim, block_dim, 0, CUDAStream::getCurrentStream()>>>
            (input, output, size, dimsize, stride);
    } else if (dimsize > 15) {
        int BLOCK_DIM_x = 16;
        int BLOCK_DIM_y = 64;
        int num_block_x = (num_blocks + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        _warpSoftmaxKernel<half, 16, 64, 2>
            <<<grid_dim, block_dim, 0, CUDAStream::getCurrentStream()>>>
            (input, output, size, dimsize, stride);
    } else if (dimsize > 7) {
        int BLOCK_DIM_x = 8;
        int BLOCK_DIM_y = 128;
        int num_block_x = (num_blocks + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        _warpSoftmaxKernel<half, 8, 128, 2>
            <<<grid_dim, block_dim, 0, CUDAStream::getCurrentStream()>>>
            (input, output, size, dimsize, stride);
    } else {
        int BLOCK_DIM_x = 4;
        int BLOCK_DIM_y = 256;
        int num_block_x = (num_blocks + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        _warpSoftmaxKernel<half, 4, 256, 2>
            <<<grid_dim, block_dim, 0, CUDAStream::getCurrentStream()>>>
            (input, output, size, dimsize, stride);
    }
}
} // namespace infini
