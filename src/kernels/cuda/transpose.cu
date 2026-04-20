#include "core/common.h"
#include "cuda/cuda_common.h"
#include "cuda/cuda_utility.h"
#include "utils/small_array.h"

constexpr unsigned int num_threads() { return 32 * 4; }
constexpr int thread_work_size() { return 4; }
constexpr int block_work_size() { return thread_work_size() * num_threads(); }

template <class T>
__global__ void _transpose_kernel(void *input, void *output, int nDims,
                                  int size, infini::SmallArray strides,
                                  infini::SmallArray outputShape) {
    int outputIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outputIdx >= size)
        return;

    int inputIdx = 0;

    if (nDims == 5) {
        // 优化版展开逻辑
        int v = outputIdx;

        int i4 = v % outputShape.data[4];
        v /= outputShape.data[4];

        int i3 = v % outputShape.data[3];
        v /= outputShape.data[3];

        int i2 = v % outputShape.data[2];
        v /= outputShape.data[2];

        int i1 = v % outputShape.data[1];
        v /= outputShape.data[1];

        int i0 = v; // 最后一维不需要 mod

        inputIdx = i0 * strides.data[0] + i1 * strides.data[1] +
                   i2 * strides.data[2] + i3 * strides.data[3] +
                   i4 * strides.data[4];
    } else if (nDims == 4) {
        int v = outputIdx;

        int i3 = v % outputShape.data[3];
        v /= outputShape.data[3];

        int i2 = v % outputShape.data[2];
        v /= outputShape.data[2];

        int i1 = v % outputShape.data[1];
        v /= outputShape.data[1];

        int i0 = v; // 最后一维不需要 mod

        inputIdx = i0 * strides.data[0] + i1 * strides.data[1] +
                   i2 * strides.data[2] + i3 * strides.data[3];
    } else {
        // fallback 通用逻辑
        int v = outputIdx;
        for (int i = nDims - 1; i >= 0; --i) {
            inputIdx += v % outputShape.data[i] * strides.data[i];
            v /= outputShape.data[i];
        }
    }
    ((T *)output)[outputIdx] = ((T *)input)[inputIdx];
}
#define CASE(T)                                                                \
    _transpose_kernel<DT_CUDA<T>::t>                                           \
        <<<gridsize, blocksize, 0, CUDAStream::getCurrentStream()>>>(          \
            input, output, nDims, size, strides, outputShape);

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

__global__ void _transpose_nchw2nhcw(const float *__restrict__ input,
                                     float *__restrict__ output, int N, int C,
                                     int H, int W) {
    using Vec4 = float4;
    int total = N * C * H * W / 4; // 每4个float一组

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;

    // 元素编号（按 float4 粒度）
    int w4 = idx % (W / 4); // W是float数，所以float4=4个float
    int c = (idx / (W / 4)) % C;
    int h = (idx / (W / 4 * C)) % H;
    int n = idx / (W / 4 * C * H);

    // input 是 [N, C, H, W]，连续存储，float4 地址对齐
    int inputOffset = (((n * C + c) * H + h) * W + w4 * 4);
    int outputOffset = (((n * H + h) * C + c) * W + w4 * 4);

    const Vec4 *input_v4 = reinterpret_cast<const Vec4 *>(input);
    Vec4 *output_v4 = reinterpret_cast<Vec4 *>(output);

    // 注意 index 是按 float4 编号，所以除以4
    output_v4[outputOffset / 4] = input_v4[inputOffset / 4];
}

namespace infini {
void transpose_kernel(int dType, void *input, void *output, int nDims, int size,
                      SmallArray strides, SmallArray outputShape) {
    int blocksize = block_work_size();
    int gridsize = (size + block_work_size() - 1) / block_work_size();
    SWITCH_DTYPE(dType)
}
void transpose_nchw2nhcw(void *input, void *output, int N, int C, int H,
                         int W) {
    int blocksize = block_work_size();
    int size = N * C * H * W;
    int gridsize = (size + 4 * block_work_size() - 1) / (4 * block_work_size());
    _transpose_nchw2nhcw<<<gridsize, blocksize, 0,
                           CUDAStream::getCurrentStream()>>>(
        (float *)input, (float *)output, N, C, H, W);
}

} // namespace infini
