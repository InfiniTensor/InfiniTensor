#pragma once
#include "core/runtime.h"
#include <cstddef>

namespace infini {

/**
 * @brief
 * 一个相比于lazy_allocator简化的allocator，管理一个大的、预先分配的内存池。
 *
 * 内存布局:
 * |<-- Weight -->|<-- IO -->|<-- Other -->|... 空闲 ...|<-- Heap -->|
 * 0          weightPeak weightPeak+ioPeak                      memPoolSize
 */
class MemoryAllocator {
  private:
    Runtime runtime;
    size_t alignment;

    // 各区域的峰值大小
    size_t weightPeak = 0;
    size_t ioPeak = 0;
    size_t peak = 0; // 中间张量的内存峰值 (由 OccamyPlanner 计算)
    size_t heapPeak = 0;

    // 内存池
    bool hasMemPool = false;
    size_t memPoolSize = 0;
    void *memPoolPtr = nullptr;

  public:
    MemoryAllocator(Runtime runtime);
    ~MemoryAllocator();

    // 设置并分配整个内存池
    void setMemPool(size_t memPoolSize);
    bool getMemPoolStatus() const;

    // 设置由 OccamyPlanner 计算的中间张量峰值
    void setPeak(size_t peak);

    // 模拟分配：仅增加区域大小并返回偏移量
    size_t allocWeight(size_t size);
    size_t allocIO(size_t size);

    // Heap 分配：从内存池尾部向前分配
    size_t heapAlloc(size_t size);
    void freeHeap();

    // 获取各区域的基地址指针
    void *getWeightPtr();
    void *getIOPtr();
    void *getPtr(); // 获取 Other 区域的基地址
    void *getHeapPtr();

    // 工具函数
    size_t getAlignedSize(size_t size);
    void info() const;
};

} // namespace infini
