#pragma once

#include <array>
#include <cstddef>

namespace infini {
// Broadcast* 保留广播索引；NoBroadcast* 要求两个输入与输出同形状。
// Base 使用 C++ 乘法；BroadcastPair/NoBroadcastUnaligned 使用 __hmul；
// NoBroadcastHalf2 使用 __hmul2。Unaligned 表示 auto 中的非对齐回退角色。
enum class Fp16MulMode {
    Base,
    BroadcastPair,
    NoBroadcastUnaligned,
    NoBroadcastHalf2,
    Auto
};
enum class Fp32MulMode { Broadcast, NoBroadcast, Auto };

// 解析接受语义名称及旧别名，空值选择 Auto；名称函数返回语义名称。
const char *mulModeName(Fp16MulMode mode);
const char *mulModeName(Fp32MulMode mode);
Fp16MulMode parseFp16MulMode(const char *value);
Fp32MulMode parseFp32MulMode(const char *value);

// 返回实际执行的分支。缓冲区要求密集存储，允许输出与整个输入共用地址，
// 不支持部分重叠或输出与被广播的输入共用地址。
Fp16MulMode dispatchMul(Fp16MulMode requested, void *a, void *b, void *c,
                        const std::array<int, 4> &aDims,
                        const std::array<int, 4> &bDims,
                        const std::array<int, 4> &cDims);
Fp32MulMode dispatchMul(Fp32MulMode requested, void *a, void *b, void *c,
                        const std::array<int, 4> &aDims,
                        const std::array<int, 4> &bDims,
                        const std::array<int, 4> &cDims);

void div_kernel(int dtypeIndex, void *a, void *b, void *c, int a0, int a1,
                int a2, int a3, int b0, int b1, int b2, int b3, int c0, int c1,
                int c2, int c3);
void add_kernel(int dtypeIndex, void *a, void *b, void *c, int a0, int a1,
                int a2, int a3, int b0, int b1, int b2, int b3, int c0, int c1,
                int c2, int c3);
void pow_kernel(int dtypeIndex, void *a, void *b, void *c, int a0, int a1,
                int a2, int a3, int b0, int b1, int b2, int b3, int c0, int c1,
                int c2, int c3);
void less_kernel(int dtypeIndex, void *a, void *b, void *c, int a0, int a1,
                 int a2, int a3, int b0, int b1, int b2, int b3, int c0, int c1,
                 int c2, int c3);
void sub_kernel(int dtypeIndex, void *a, void *b, void *c, int a0, int a1,
                int a2, int a3, int b0, int b1, int b2, int b3, int c0, int c1,
                int c2, int c3);
void mul_kernel(int dtypeIndex, void *a, void *b, void *c, int a0, int a1,
                int a2, int a3, int b0, int b1, int b2, int b3, int c0, int c1,
                int c2, int c3);
void min_kernel(int dtypeIndex, void *a, void *b, void *c, int a0, int a1,
                int a2, int a3, int b0, int b1, int b2, int b3, int c0, int c1,
                int c2, int c3);
void max_kernel(int dtypeIndex, void *a, void *b, void *c, int a0, int a1,
                int a2, int a3, int b0, int b1, int b2, int b3, int c0, int c1,
                int c2, int c3);

void div_const_kernel(int dType, void *a, void *b, void *c, std::size_t n);

void pow_const_kernel(int dType, void *a, void *b, void *c, std::size_t n);
}; // namespace infini
