#include "cuda/cuda_common.h"
#include "cuda/cuda_utility.h"

// 每个线程处理一个 (outer, inner) 对应的位置，在 axis 维上遍历 axis_size 个元素
template <class T>
__global__ void _argmax_kernel(void *input, int64_t *output, int outer,
                               int inner, int axis_size,
                               int select_last_index) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * inner;
    if (idx >= total)
        return;

    // 计算 outer / inner 坐标
    int o = idx / inner;
    int i = idx % inner;

    // 定位第一个元素: input + o * axis_size * inner + 0 * inner + i
    T *base =
        (T *)input + (size_t)o * (size_t)axis_size * (size_t)inner + (size_t)i;

    // 初始化为第 0 个元素
    T max_val = base[0 * inner];
    int64_t max_idx = 0;

    // 从 a = 1 开始比较
    for (int a = 1; a < axis_size; ++a) {
        T val = base[(size_t)a * (size_t)inner];
        if (select_last_index) {
            // 选择“最后一个最大值”：遇到相等也更新
            if (val >= max_val) {
                max_val = val;
                max_idx = a;
            }
        } else {
            // 默认选择第一个最大值：只有严格更大才更新
            if (val > max_val) {
                max_val = val;
                max_idx = a;
            }
        }
    }

    // 写入输出。
    output[idx] = max_idx;
}
namespace infini {

#define CASE(OP, T)                                                            \
    _##OP##_kernel<DT_CUDA<T>::t>                                              \
        <<<blocks, threads, 0, CUDAStream::getCurrentStream()>>>(              \
            input, output, outer, inner, axis_size, select_last_index);

#define SWITCH_DTYPE(OP, DTYPE)                                                \
    switch (DTYPE) {                                                           \
    case 1:                                                                    \
        CASE(OP, 1)                                                            \
        break;                                                                 \
    case 2:                                                                    \
        CASE(OP, 2)                                                            \
        break;                                                                 \
    case 3:                                                                    \
        CASE(OP, 3)                                                            \
        break;                                                                 \
    case 4:                                                                    \
        CASE(OP, 4)                                                            \
        break;                                                                 \
    case 5:                                                                    \
        CASE(OP, 5)                                                            \
        break;                                                                 \
    case 6:                                                                    \
        CASE(OP, 6)                                                            \
        break;                                                                 \
    case 7:                                                                    \
        CASE(OP, 7)                                                            \
        break;                                                                 \
    case 10:                                                                   \
        CASE(OP, 10)                                                           \
        break;                                                                 \
    case 11:                                                                   \
        CASE(OP, 11)                                                           \
        break;                                                                 \
    case 12:                                                                   \
        CASE(OP, 12)                                                           \
        break;                                                                 \
    case 13:                                                                   \
        CASE(OP, 13)                                                           \
        break;                                                                 \
    case 16:                                                                   \
        CASE(OP, 16)                                                           \
        break;                                                                 \
    default:                                                                   \
        IT_TODO_HALT();                                                        \
    }

void argmax_kernel(void *input, int64_t *output, int outer, int inner,
                   int axis_size, int select_last_index, int dType) {
    int total = outer * inner;
    if (total <= 0)
        return;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    SWITCH_DTYPE(argmax, dType)
}
} // namespace infini