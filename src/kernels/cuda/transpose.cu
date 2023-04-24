#include "core/common.h"
#include "cuda/cuda_common.h"
#include "utils/small_array.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <limits>
#define CUDA_HOST_DEVICE __forceinline__ __device__ __host__

// https://github.com/462630221/SampleCode
template <typename T> struct QuotientMod {
    T quotient;
    T mod;
    __host__ __device__ QuotientMod(T q, T m) : quotient(q), mod(m) {}
};

template <typename T> struct FastIntDivider {
    FastIntDivider() {}
    FastIntDivider(T d) { divisor_ = d; };
    __forceinline__ __device__ __host__ T div(T n) { return n / divisor_; }
    __forceinline__ __device__ __host__ T mod(T n) { return n % divisor_; }
    __forceinline__ __device__ __host__ QuotientMod<T> divmod(T n) {
        return QuotientMod<T>(n / divisor_, n % divisor_);
    }
    T divisor_;
};

template <> struct FastIntDivider<uint32_t> {
    FastIntDivider(){};

    FastIntDivider(uint32_t d) {
        assert(d >= 1);
        divisor_ = d;
        // if put 0 to __builtin_clz, the result undefined.
        if (d == 1) {
            rshift_ = 0;
        } else {
            rshift_ = 32 - __builtin_clz(d - 1);
        }
        uint64_t magic_t = ((1lu << (32 + rshift_)) + d - 1) / d;
        magic_ = uint32_t(magic_t);
    };

    __forceinline__ __device__ __host__ uint32_t div(uint32_t n) {
#if defined(__CUDA_ARCH__)
        uint32_t q = __umulhi(n, magic_);
#else
        uint32_t q = (uint64_t(n) * magic_) >> 32;
#endif
        // return (((n - q) >> 1) + q) >> (rshift_ - 1);
        return (n + q) >> rshift_;
    }

    __forceinline__ __device__ __host__ QuotientMod<uint32_t>
    divmod(uint32_t n) {
        uint32_t q = div(n);
        return QuotientMod<uint32_t>(q, n - divisor_ * q);
    }

    uint32_t magic_;
    uint32_t rshift_;
    uint32_t divisor_;
};

void test_fast_u32() {
    uint32_t d = 1;

    FastIntDivider<uint32_t> diver(d);
    std::cout << "7/3= " << uint32_t(7) / uint32_t(d) << " " << diver.div(7)
              << std::endl;
}

constexpr unsigned int num_threads() { return 32 * 4; }
constexpr int thread_work_size() { return 4; }
constexpr int block_work_size() { return thread_work_size() * num_threads(); }

__global__ void _transpose_kernel(float *input, float *output, int nDims,
                                  int size, infini::SmallArray strides,
                                  infini::SmallArray outputShape) {
    int outputIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outputIdx < size) {
        int inputIdx = 0;
        int v = outputIdx;
        for (int i = nDims - 1; i >= 0; --i) {
            inputIdx += v % outputShape.data[i] * strides.data[i];
            v /= outputShape.data[i];
        }
#if __CUDA_ARCH__ >= 350 || defined(USE_ROCM)
        output[outputIdx] = __ldg(input + inputIdx);
#else
        output[outputIdx] = input[inputIdx];
#endif
    }
}

template <typename T, int NUM> struct Array {
    CUDA_HOST_DEVICE T &operator[](unsigned int index) { return data[index]; }
    CUDA_HOST_DEVICE const T &operator[](unsigned int index) const {
        return data[index];
    }
    CUDA_HOST_DEVICE constexpr int size() const { return NUM; }

    CUDA_HOST_DEVICE Array() {
#ifndef __CUDA_ARCH__
        for (int i = 0; i < NUM; i++) {
            data[i] = T();
        }
#endif
    }

    T data[NUM];
};

/**
 * @brief Optimize : Reorganize
 *
 */
template <int NUM_AXES, int UNROLL, int BLOCK_SIZE, typename T>
__global__ void
transpose_kernel_v3(const T *data_in, T *data_out,
                    const Array<uint32_t, NUM_AXES> perm_strides,
                    Array<FastIntDivider<uint32_t>, NUM_AXES> out_strides,
                    const size_t all_cnt) {
    uint32_t out_offset_reg[UNROLL];
    uint32_t in_offset_reg[UNROLL];
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        out_offset_reg[i] =
            blockIdx.x * BLOCK_SIZE * UNROLL + threadIdx.x + i * BLOCK_SIZE;
        in_offset_reg[i] = 0;
    }

#pragma unroll
    for (int j = 0; j < NUM_AXES; ++j) {
#pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            QuotientMod<uint32_t> d = out_strides[j].divmod(out_offset_reg[i]);
            in_offset_reg[i] += d.quotient * perm_strides[j];
            out_offset_reg[i] = d.mod;
        }
    }

    T ld_reg[UNROLL];
    uint32_t offset = blockIdx.x * BLOCK_SIZE * UNROLL + threadIdx.x;
    if (offset + BLOCK_SIZE * UNROLL <= all_cnt) {
#pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            ld_reg[i] = data_in[in_offset_reg[i]];
        }
#pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            data_out[offset + i * BLOCK_SIZE] = ld_reg[i];
        }
    } else {
#pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (offset + i * BLOCK_SIZE < all_cnt) {
                ld_reg[i] = data_in[in_offset_reg[i]];
            }
        }
#pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (offset + i * BLOCK_SIZE < all_cnt) {
                data_out[offset + i * BLOCK_SIZE] = ld_reg[i];
            }
        }
    }
}



template <typename T> T AccMul(std::vector<T> vec) {
    return std::accumulate(vec.begin(), vec.end(), T(1), std::multiplies<T>());
}

namespace infini {
// void transpose_kernel(float *input, float *output, int nDims, int size,
//                       SmallArray strides, SmallArray outputShape) {
//     int blocksize = block_work_size();
//     int gridsize = (size + block_work_size() - 1) / block_work_size();
//     _transpose_kernel<<<gridsize, blocksize>>>(input, output, nDims, size,
//                                                strides, outputShape);
// }

std::vector<uint32_t> GetStrides(std::vector<uint32_t> dims) {
    std::vector<uint32_t> strides(dims.size(), 1);
    for (int i = dims.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    return strides;
}

void transpose_kernel(float *input, float *output, int nDims, int size,
                      SmallArray _strides, SmallArray _outputShape,
                      vector<int> _dims_in, vector<int> _dims_out,
                      vector<int> _perms) {
    constexpr int NUM_AXES = 4;
    IT_ASSERT(nDims <= NUM_AXES);
    constexpr int UNROLL = 8 / sizeof(float);
    constexpr int BLOCK_SIZE = 128;

    vector<uint32_t> dims_in, dims_out, perms;
    for (auto v : _dims_in)
        dims_in.push_back(v);
    for (auto v : _dims_out)
        dims_out.push_back(v);
    for (auto v : _perms)
        perms.push_back(v);

    size_t all_cnt = AccMul(dims_in);

    auto strides_in = GetStrides(dims_in);
    auto strides_out = GetStrides(dims_out);

    const int grid =
        (all_cnt + BLOCK_SIZE * UNROLL - 1) / (BLOCK_SIZE * UNROLL);
    Array<uint32_t, NUM_AXES> perm_strides;
    Array<FastIntDivider<uint32_t>, NUM_AXES> out_strides_fast;
    for (int i = 0; i < NUM_AXES; ++i) {
        out_strides_fast[i] = FastIntDivider<uint32_t>(strides_out[i]);
        perm_strides[i] = strides_in[perms[i]];
    }

    transpose_kernel_v3<NUM_AXES, UNROLL, BLOCK_SIZE, float>
        <<<grid, BLOCK_SIZE, 0>>>(
            input, output, perm_strides, out_strides_fast, all_cnt);
}

} // namespace infini
