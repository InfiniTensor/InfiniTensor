#pragma once
#include "core/graph.h"
#include "core/tensor.h"
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace infini {

struct OccamyFreeBlock {
    size_t start; // 起始偏移（字节）
    size_t end;   // 结束偏移（字节，exclusive）
};

struct OccamyAllocBlock {
    TensorObj *tensorPtr;
    size_t start;
    size_t end; // exclusive
};

struct TensorLiveness {
    TensorObj *tensor;
    size_t sizeBytes;
    int firstUseStep; // 第一次被使用的 op 序号（topo 序）
    int lastUseStep;  // 最后一次被使用的 op 序号
};

/**
 * 移植自"Occamy: Memory-efficient GPU Compiler for DNN Inference" from DAC 2023
 */
class OccamyPlanner {
  public:
    enum class Policy {
        FIRST_FIT,        // 从低地址找第一个能放下的空闲块
        BEST_FIT,         // 找最小的能放下的空闲块
        LONGER_FIRST_FIT, // 按 liveness 长度降序，用 first-fit 放置
        BIGGER_FIRST_FIT, // 按 tensor 大小降序，用 first-fit 放置
    };

    /**
     * @param ops           拓扑排序后的算子列表
     * @param tensorToOffset 输出：每个 intermediate tensor 对应的内存池偏移量
     * @return 最小内存池大小（字节）
     */
    static size_t plan(const OpVec &ops,
                       std::unordered_map<TensorObj *, size_t> &tensorToOffset);

  private:
    static std::vector<TensorLiveness> analyzeLiveness(const OpVec &ops);

    static int findFirstFit(const std::vector<OccamyFreeBlock> &freePool,
                            size_t size);
    static int findBestFit(const std::vector<OccamyFreeBlock> &freePool,
                           size_t size);

    static size_t allocSimul(TensorObj *tensor, size_t size,
                             std::vector<OccamyFreeBlock> &freePool,
                             std::vector<OccamyAllocBlock> &allocPool,
                             Policy fitPolicy);

    static bool deallocSimul(TensorObj *tensor,
                             std::vector<OccamyFreeBlock> &freePool,
                             std::vector<OccamyAllocBlock> &allocPool);

    static size_t getCurrentPeak(const std::vector<OccamyFreeBlock> &freePool);
    static size_t
    runSchedule(const std::vector<TensorLiveness> &liveness, const OpVec &ops,
                Policy policy,
                std::unordered_map<TensorObj *, size_t> &tensorToOffset);

    static size_t alignUp(size_t size, size_t alignment = 256);
};

} // namespace infini
