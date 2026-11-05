#pragma once
#include <cstdio>
#include <cuda_fp16.h>

const int BATCH_SIZE = 32; // 单次内核最多处理的分段数，与模型的 batch 维无关。
const int DIM_MAX_SIZE = 8; // 元数据支持的最大张量维数。

// 小张量（分段）的元数据。Split 将大张量写入这些分段，Concat 则反向读取。
// 各分段按连续存储访问；只有大张量侧通过 stride 描述实际布局。
template <typename T> struct ElementTensorMetadata {
    T *data[BATCH_SIZE];       // 各分段的设备地址。
    int dimBgNo[BATCH_SIZE];   // 各分段在拆分/拼接轴上的起始坐标。
    int dimSize[BATCH_SIZE];   // 各分段在该轴上的长度。
    int nElements[BATCH_SIZE]; // 各分段的标量元素总数，打包前后含义不变。
    void print() const {
        for (int i = 0; i < BATCH_SIZE; i++)
            printf("%d:(data=%p,dimBgNo=%d,dimSize=%d,nElements=%d)\n", i,
                   data[i], dimBgNo[i], dimSize[i], nElements[i]);
    }
};
// 大张量的形状、步长及设备地址；stride 的单位是 T 元素，而非字节。
template <typename T> struct ComposedTensorMetadata {
    int dimSize[DIM_MAX_SIZE]; // 各维度长度。
    int stride[DIM_MAX_SIZE];  // 各维度坐标增加 1 时，地址跨过的元素数。
    T *data;
};

namespace infini {

// FP16 的复制策略。所有分支只搬运数据，不进行浮点计算。
// Auto：密集且可打包时走 32 位行复制；密集但不可打包时走 16 位行复制；
// 非密集布局回退原始 half 复制。32 位打包表示搬运两个 half，并非转成 FP32。
enum class Fp16CopyMode {
    Auto,                  // 根据布局和对齐条件选择实际路径。
    StridedHalf,           // 通用逐维索引，逐个复制 half，兼作基线和回退。
    ContiguousRowBits16,   // 密集布局使用折叠行索引，逐个复制 16 位原始数据。
    ContiguousRowPacked32  // 折叠行索引，一次复制两个 half 的 32 位原始数据。
};

// 模式名称用于日志；解析函数接受保留模式及兼容别名，空配置默认 Auto。
const char *copyModeName(Fp16CopyMode mode);
Fp16CopyMode parseFp16CopyMode(const char *value);
// 校验元数据、启动复制并返回实际路径；全空任务不启动内核，返回请求模式。
// batchSize 是本次传入的分段数；isSplit=true 表示大张量到分段，false 表示反向。
Fp16CopyMode dispatchCopy(Fp16CopyMode requested,
                          const ElementTensorMetadata<half> &eleMeta,
                          const ComposedTensorMetadata<half> &compMeta, int dim,
                          int batchSize, int nDims, bool isSplit);

// FP32 只切换索引方式，复制单位始终是单个 float。
// Auto 对密集布局使用行索引，非密集布局使用原通用索引。
enum class Fp32CopyMode { StridedScalar, ContiguousRowScalar, Auto };

const char *copyModeName(Fp32CopyMode mode);
Fp32CopyMode parseFp32CopyMode(const char *value);
// 返回实际选择的路径；即使任务全空，也返回已解析的具体路径。
Fp32CopyMode dispatchCopy(Fp32CopyMode requested,
                          const ElementTensorMetadata<float> &eleMeta,
                          const ComposedTensorMetadata<float> &compMeta,
                          int dim, int batchSize, int nDims, bool isSplit);

// CUDA/CoreX/MetaX 共用入口，按对应精度的环境变量分派，默认 Auto。
void split_concat_kernel(const ElementTensorMetadata<float> &eleMeta,
                         const ComposedTensorMetadata<float> &compMeta, int dim,
                         int batchSize, int nDims, bool isSplit);
void split_concat_kernel(const ElementTensorMetadata<half> &eleMeta,
                         const ComposedTensorMetadata<half> &compMeta, int dim,
                         int batchSize, int nDims, bool isSplit);
} // namespace infini
