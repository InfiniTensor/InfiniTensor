#include "cuda/cuda_common.h"

#include "cuda/cuda_split_concat.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

// 通用索引：将连续小张量的一维编号映射到大张量的实际元素偏移。
// dimBgNo/dimSize 分别是分段在拆分轴上的起点和长度，dim 是拆分轴。
// 逐维使用大张量 stride，因此大张量侧存在 padding 时仍可正确寻址。
template <typename T>
__host__ __device__ int
stridedElementIndexToComposedIndex(int elementIndex, int dimBgNo, int dimSize,
                                   int dim, int nDim,
                                   ComposedTensorMetadata<T> wholeMeta) {
    int offset = 0;

#pragma unroll
    for (int i = nDim - 1; i >= 1; --i) {
        // 从低维到高维还原小张量坐标；拆分轴的长度使用当前分段长度。
        int size = (i == dim) ? dimSize : wholeMeta.dimSize[i];
        int p = elementIndex % size;
        // 仅在拆分轴加上分段起点，其余维度的坐标与大张量相同。
        int oP = (i == dim) ? (p + dimBgNo) : p;
        elementIndex = (elementIndex - p) / size;
        offset += oP * wholeMeta.stride[i];
    }

    // 剩余编号就是第 0 维坐标，无需再做一次除法和取模。
    int oP = (dim == 0) ? (elementIndex + dimBgNo) : elementIndex;
    return offset + oP * wholeMeta.stride[0];
}

// 通用标量复制内核：blockIdx.y 选择分段，x 方向的线程处理分段内元素。
// isSplit=true：大张量复制到分段；false：分段复制到大张量。
template <typename T>
__global__ void
split_concat_strided_scalar_copy_kernel(ElementTensorMetadata<T> elemMeta,
                                        ComposedTensorMetadata<T> compMeta,
                                        int dim, int nDims, bool isSplit) {
    int nElements = elemMeta.nElements[blockIdx.y];
    int dimBgNo = elemMeta.dimBgNo[blockIdx.y];
    int dimSize = elemMeta.dimSize[blockIdx.y];
    T *elemData = elemMeta.data[blockIdx.y];
    // 此 stride 是线程循环的步长，不是张量的维度 stride。
    int stride = gridDim.x * blockDim.x;

    // 限制 grid.x 后，每个线程继续按整个 x 网格的线程数步进，避免漏拷贝。
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < nElements;
         tid += stride) {
        int offset = stridedElementIndexToComposedIndex<T>(
            tid, dimBgNo, dimSize, dim, nDims, compMeta);

        if (isSplit) {
            elemData[tid] = compMeta.data[offset];
        } else {
            compMeta.data[offset] = elemData[tid];
        }
    }
}

namespace infini {

namespace {

// 将日志标准名称和旧配置别名集中维护，供 FP16/FP32 共用解析逻辑。
template <typename Mode> struct CopyModeEntry {
    Mode mode;
    const char *canonical;
    const char *legacyAlias;
};

constexpr CopyModeEntry<Fp16CopyMode> fp16CopyModes[]{
    {Fp16CopyMode::Auto, "auto", nullptr},
    {Fp16CopyMode::StridedHalf, "strided_half", "base"},
    {Fp16CopyMode::ContiguousRowBits16, "contiguous_row_bits16", "row16"},
    {Fp16CopyMode::ContiguousRowPacked32, "contiguous_row_packed32", "pack32"}};
constexpr CopyModeEntry<Fp32CopyMode> fp32CopyModes[]{
    {Fp32CopyMode::StridedScalar, "strided_scalar", "base"},
    {Fp32CopyMode::ContiguousRowScalar, "contiguous_row_scalar", "row"},
    {Fp32CopyMode::Auto, "auto", nullptr}};

template <typename Mode, std::size_t N>
const char *modeName(Mode mode, const CopyModeEntry<Mode> (&entries)[N]) {
    for (const auto &entry : entries)
        if (mode == entry.mode)
            return entry.canonical;
    return "invalid";
}

template <typename Mode, std::size_t N>
Mode parseMode(const char *value, const CopyModeEntry<Mode> (&entries)[N],
               Mode defaultMode, const char *error) {
    // 未设置或空字符串使用默认模式；未知名称直接报错，防止静默换分支。
    if (!value || !*value)
        return defaultMode;
    for (const auto &entry : entries)
        if (std::strcmp(value, entry.canonical) == 0 ||
            (entry.legacyAlias && std::strcmp(value, entry.legacyAlias) == 0))
            return entry.mode;
    throw std::invalid_argument(error);
}

// 原始复制路径：所有分段共用 x 网格，规模按最大分段的元素数确定。
template <typename T>
void launchStridedScalarCopy(const ElementTensorMetadata<T> &eleMeta,
                             const ComposedTensorMetadata<T> &compMeta, int dim,
                             int batchSize, int nDims, bool isSplit) {
    if (batchSize <= 0)
        return;
    constexpr int threads = 512;
    const int maxElements =
        *std::max_element(eleMeta.nElements, eleMeta.nElements + batchSize);
    if (maxElements <= 0)
        return;
    // 最多启动 1024 个 x 方向 block，其余元素由内核循环完成。
    // 使用 1+(n-1)/threads 向上取整，避免 n+threads-1 的中间加法溢出。
    const int blocks = std::min(1 + (maxElements - 1) / threads, 1024);
    split_concat_strided_scalar_copy_kernel<T>
        <<<dim3(blocks, batchSize), dim3(threads), 0,
           CUDAStream::getCurrentStream()>>>(eleMeta, compMeta, dim, nDims,
                                             isSplit);
}

// 连续行复制的紧凑元数据，计数、长度和偏移的单位全部是 T。
// T 可为 uint16_t、uint32_t 或 float，分别对应 half 标量、half 打包和 FP32。
template <typename T> struct ContiguousRowCopyMetadata {
    T *elements[BATCH_SIZE];      // 各分段首地址。
    T *composed;                 // 大张量首地址。
    int elementsCount[BATCH_SIZE]; // 各分段需要复制的 T 单元数量。
    int chunk[BATCH_SIZE];       // 分段在一行内占用的连续 T 单元数量。
    int offset[BATCH_SIZE];      // 分段在大张量每行内的起始偏移。
    int composedRow;             // 大张量每行的跨度。
};

// 密集布局可以折叠为 [outer, axis, inner]：outer/inner 分别为拆分轴前/后
// 各维度长度的乘积。一行是 axis*inner 个标量，分段占其中连续的一块。
// 例如 NCHW 沿 C 拆分时，outer=N、inner=H*W。
// 只计算行号和行内偏移，省去通用内核逐维还原坐标的开销。
// T 决定复制宽度；即使 T=uint32_t，也只是搬运两个 half 的原始位。
template <typename T>
__global__ void
split_concat_contiguous_row_copy_kernel(ContiguousRowCopyMetadata<T> meta,
                                        bool isSplit) {
    const int piece = blockIdx.y;
    const int count = meta.elementsCount[piece];
    // 空分段的 chunk 可能为 0，必须在后续除法之前退出。
    if (count == 0)
        return;
    const int chunk = meta.chunk[piece];
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += stride) {
        // i 是小张量内的复制单元编号；i-row*chunk 是该单元在本行内的位置。
        const int row = i / chunk;
        const int offset =
            row * meta.composedRow + meta.offset[piece] + (i - row * chunk);
        if (isSplit)
            meta.elements[piece][i] = meta.composed[offset];
        else
            meta.composed[offset] = meta.elements[piece][i];
    }
}

// 构造行复制元数据并启动内核；复制单位由源类型和访问类型的大小决定。
// 调用方已检查可整除及地址对齐；reinterpret_cast 只改变访问单位，不转换数值。
template <typename T, typename Source>
void launchContiguousRowCopy(const ElementTensorMetadata<Source> &eleMeta,
                             const ComposedTensorMetadata<Source> &compMeta,
                             int batchSize, int inner, int axis,
                             int maxScalarElements, bool isSplit) {
    constexpr int unit = sizeof(T) / sizeof(Source);
    ContiguousRowCopyMetadata<T> meta{};
    meta.composed = reinterpret_cast<T *>(compMeta.data);
    meta.composedRow = int(int64_t(compMeta.dimSize[axis]) * inner / unit);
    for (int i = 0; i < batchSize; ++i) {
        meta.elements[i] = reinterpret_cast<T *>(eleMeta.data[i]);
        meta.elementsCount[i] = eleMeta.nElements[i] / unit;
        meta.chunk[i] = int(int64_t(eleMeta.dimSize[i]) * inner / unit);
        meta.offset[i] = int(int64_t(eleMeta.dimBgNo[i]) * inner / unit);
    }
    // 按打包前的标量元素数确定网格，使 16 位和 32 位行复制使用相同启动规模。
    // 打包分支的复制单元数减半，每次迭代搬运两个 half。
    const int blocks = std::min(1 + (maxScalarElements - 1) / 512, 1024);
    split_concat_contiguous_row_copy_kernel<T>
        <<<dim3(blocks, batchSize), dim3(512), 0,
           CUDAStream::getCurrentStream()>>>(meta, isSplit);
}

// 主机侧校验结果；仅用于选路，不读取设备缓冲区中的数据。
struct CopyMetadata {
    int maxElements = 0; // 最大非空分段的标量元素数，决定启动规模。
    int inner = 1;       // 拆分轴之后各维度长度的乘积。
    bool dense = true;   // 大张量是否具有标准连续步长。
    bool packed = true;  // 大张量和所有非空分段是否满足 32 位打包条件。
};

// 校验维度、索引范围、分段边界和指针对齐，汇总行索引所需信息。
template <typename T>
CopyMetadata validateCopy(const ElementTensorMetadata<T> &eleMeta,
                          const ComposedTensorMetadata<T> &compMeta, int dim,
                          int batchSize, int nDims) {
    constexpr bool isHalf = std::is_same_v<T, half>;
    const auto fail = [=](const char *reason) {
        throw std::invalid_argument(
            std::string(isHalf ? "FP16 copy: " : "FP32 copy: ") + reason);
    };
    if (batchSize < 0 || batchSize > BATCH_SIZE || nDims < 1 ||
        nDims > DIM_MAX_SIZE || dim < 0 || dim >= nDims)
        fail("invalid metadata rank");
    CopyMetadata result;
    // product 是当前维度应有的连续步长；maxOffset 是大张量最大元素偏移。
    // 中间计算用 64 位，最终仍限制在内核 int 索引可安全表示的范围内。
    int64_t product = 1, maxOffset = 0, inner = 1, outer = 1;
    for (int d = nDims - 1; d >= 0; --d) {
        if (compMeta.dimSize[d] < 0 || compMeta.stride[d] < 0)
            fail("invalid dimension");
        result.dense = result.dense && compMeta.stride[d] == product;
        maxOffset +=
            int64_t(std::max(0, compMeta.dimSize[d] - 1)) * compMeta.stride[d];
        product *= compMeta.dimSize[d];
        if (product > std::numeric_limits<int>::max() ||
            maxOffset >= std::numeric_limits<int>::max())
            fail("int index overflow");
        if (d > dim)
            inner *= compMeta.dimSize[d];
        if (d < dim) {
            outer *= compMeta.dimSize[d];
            if (outer > std::numeric_limits<int>::max())
                fail("outer index overflow");
        }
    }
    result.inner = int(inner);
    // 两个 half 组成一个 32 位单元：首地址须 4 字节对齐，行跨度须为偶数个
    // half。
    if constexpr (isHalf)
        result.packed =
            reinterpret_cast<std::uintptr_t>(compMeta.data) % 4 == 0 &&
            (int64_t(compMeta.dimSize[dim]) * inner) % 2 == 0;
    for (int p = 0; p < batchSize; ++p) {
        // 分段必须落在拆分轴范围内，元素总数应等于 outer*分段轴长*inner。
        // 元素数上界还为线程循环的下一次 int 步进预留空间。
        if (eleMeta.nElements[p] < 0 ||
            eleMeta.nElements[p] >
                std::numeric_limits<int>::max() - 512 * 1024 ||
            eleMeta.dimSize[p] < 0 || eleMeta.dimBgNo[p] < 0 ||
            int64_t(eleMeta.dimBgNo[p]) + eleMeta.dimSize[p] >
                compMeta.dimSize[dim] ||
            eleMeta.nElements[p] != outer * eleMeta.dimSize[p] * inner)
            fail("invalid piece");
        result.maxElements = std::max(result.maxElements, eleMeta.nElements[p]);
        if (eleMeta.nElements[p]) {
            if (!eleMeta.data[p] || !compMeta.data ||
                reinterpret_cast<std::uintptr_t>(eleMeta.data[p]) %
                    alignof(T) ||
                reinterpret_cast<std::uintptr_t>(compMeta.data) % alignof(T))
                fail(isHalf ? "null/unaligned half pointer"
                            : "null/unaligned float pointer");
            // 每个非空分段的首地址、行块长度和行内起点都必须能按 32 位访问。
            // 任一分段不满足时，整次行复制回退 16 位，避免跨分段读取。
            if constexpr (isHalf)
                result.packed =
                    result.packed &&
                    reinterpret_cast<std::uintptr_t>(eleMeta.data[p]) % 4 ==
                        0 &&
                    (int64_t(eleMeta.dimSize[p]) * inner) % 2 == 0 &&
                    (int64_t(eleMeta.dimBgNo[p]) * inner) % 2 == 0;
        }
    }
    // 非空任务不能使用零步长；全空任务允许空指针且不会启动内核。
    if (result.maxElements)
        for (int d = 0; d < nDims; ++d)
            if (compMeta.stride[d] == 0)
                fail("zero nonempty stride");
    return result;
}
} // namespace

const char *copyModeName(Fp16CopyMode mode) {
    return modeName(mode, fp16CopyModes);
}

const char *copyModeName(Fp32CopyMode mode) {
    return modeName(mode, fp32CopyModes);
}

Fp16CopyMode parseFp16CopyMode(const char *value) {
    return parseMode(
        value, fp16CopyModes, Fp16CopyMode::Auto,
        "INFINITENSOR_FP16_COPY_MODE: expected "
        "auto|strided_half|contiguous_row_bits16|contiguous_row_packed32 "
        "(legacy aliases: base|row16|pack32); "
        "strided_bits16/bits16, strided_bits16_pair/generic_pair and "
        "strided_packed32/generic_pack32 were removed");
}

Fp32CopyMode parseFp32CopyMode(const char *value) {
    return parseMode(
        value, fp32CopyModes, Fp32CopyMode::Auto,
        "INFINITENSOR_FP32_COPY_MODE: expected "
        "auto|strided_scalar|contiguous_row_scalar (legacy aliases: base|row)");
}

// FP16 分派：保留原始基线，并为连续布局选择已验证有效的行复制分支。
Fp16CopyMode dispatchCopy(Fp16CopyMode requested,
                          const ElementTensorMetadata<half> &eleMeta,
                          const ComposedTensorMetadata<half> &compMeta, int dim,
                          int batchSize, int nDims, bool isSplit) {
    if (requested < Fp16CopyMode::Auto ||
        requested > Fp16CopyMode::ContiguousRowPacked32)
        throw std::invalid_argument("FP16 copy: invalid mode");
    const auto meta = validateCopy(eleMeta, compMeta, dim, batchSize, nDims);
    // 全空任务无需复制，保留请求模式作为返回值。
    if (meta.maxElements == 0)
        return requested;
    // 显式请求基线，或大张量非连续时，使用支持 stride 的通用 half 复制。
    if (requested == Fp16CopyMode::StridedHalf || !meta.dense) {
        launchStridedScalarCopy(eleMeta, compMeta, dim, batchSize, nDims,
                                isSplit);
        return Fp16CopyMode::StridedHalf;
    }

    // Auto 和显式行打包模式在满足全部对齐条件时，一次搬运两个 half。
    if (requested != Fp16CopyMode::ContiguousRowBits16 && meta.packed) {
        launchContiguousRowCopy<std::uint32_t>(
            eleMeta, compMeta, batchSize, meta.inner, dim, meta.maxElements,
            isSplit);
        return Fp16CopyMode::ContiguousRowPacked32;
    }
    // 连续但不可打包，或显式要求 16 位行复制时，保留简化索引、逐个复制 half。
    launchContiguousRowCopy<std::uint16_t>(
        eleMeta, compMeta, batchSize, meta.inner, dim, meta.maxElements,
        isSplit);
    return Fp16CopyMode::ContiguousRowBits16;
}

Fp32CopyMode dispatchCopy(Fp32CopyMode requested,
                          const ElementTensorMetadata<float> &eleMeta,
                          const ComposedTensorMetadata<float> &compMeta,
                          int dim, int batchSize, int nDims, bool isSplit) {
    if (requested < Fp32CopyMode::StridedScalar ||
        requested > Fp32CopyMode::Auto)
        throw std::invalid_argument("FP32 copy: invalid mode");
    const auto meta = validateCopy(eleMeta, compMeta, dim, batchSize, nDims);
    // Auto/显式行模式遇到非连续布局均回退；空任务也返回解析后的实际路径。
    const Fp32CopyMode actual =
        requested == Fp32CopyMode::StridedScalar || !meta.dense
            ? Fp32CopyMode::StridedScalar
            : Fp32CopyMode::ContiguousRowScalar;
    if (!meta.maxElements)
        return actual;
    if (actual == Fp32CopyMode::StridedScalar)
        launchStridedScalarCopy<float>(eleMeta, compMeta, dim, batchSize, nDims,
                                       isSplit);
    else
        launchContiguousRowCopy<float>(eleMeta, compMeta, batchSize, meta.inner,
                                       dim, meta.maxElements, isSplit);
    return actual;
}

void split_concat_kernel(const ElementTensorMetadata<float> &eleMeta,
                         const ComposedTensorMetadata<float> &compMeta, int dim,
                         int batchSize, int nDims, bool isSplit) {
    // 每个进程首次调用此精度的入口时读取一次配置；切换变量需使用新进程。
    static const auto mode =
        parseFp32CopyMode(std::getenv("INFINITENSOR_FP32_COPY_MODE"));
    if (mode != Fp32CopyMode::StridedScalar) {
        dispatchCopy(mode, eleMeta, compMeta, dim, batchSize, nDims, isSplit);
        return;
    }
    // 显式选择基线时，继续使用通用 float 内核。
    launchStridedScalarCopy(eleMeta, compMeta, dim, batchSize, nDims, isSplit);
}

void split_concat_kernel(const ElementTensorMetadata<half> &eleMeta,
                         const ComposedTensorMetadata<half> &compMeta, int dim,
                         int batchSize, int nDims, bool isSplit) {
    // FP16 与 FP32 使用独立配置；未设置时默认为 Auto。
    static const auto mode =
        parseFp16CopyMode(std::getenv("INFINITENSOR_FP16_COPY_MODE"));
    if (mode != Fp16CopyMode::StridedHalf) {
        dispatchCopy(mode, eleMeta, compMeta, dim, batchSize, nDims, isSplit);
        return;
    }
    // 显式选择基线时，继续使用通用 half 内核。
    launchStridedScalarCopy(eleMeta, compMeta, dim, batchSize, nDims, isSplit);
}

} // namespace infini
