#include "core/memory_allocator.h"
#include <cstddef>
#include <iostream>

namespace infini {

constexpr size_t alignmentInBytesForCUDA = 256;

MemoryAllocator::MemoryAllocator(Runtime runtime) : runtime(runtime) {
    if (runtime->isCuda()) {
        alignment = alignmentInBytesForCUDA;
    } else {
        alignment = sizeof(uint64_t);
    }
}

MemoryAllocator::~MemoryAllocator() {
    if (this->memPoolPtr != nullptr) {
        runtime->dealloc(this->memPoolPtr);
        this->memPoolPtr = nullptr;
    }
}

void MemoryAllocator::setMemPool(size_t memPoolSize) {
    IT_ASSERT(memPoolSize > 0, "Memory pool size must be positive.");
    if (!this->hasMemPool) {
        this->hasMemPool = true;
        this->memPoolSize = memPoolSize;
        this->memPoolPtr = runtime->alloc(memPoolSize);
        IT_ASSERT(this->memPoolPtr != nullptr,
                  "Memory pool allocation failed.");
    } else {
        IT_ASSERT(this->memPoolSize == memPoolSize,
                  "Memory pool already set with a different size.");
    }
}

bool MemoryAllocator::getMemPoolStatus() const { return this->hasMemPool; }

void MemoryAllocator::setPeak(size_t peak) { this->peak = peak; }

size_t MemoryAllocator::allocWeight(size_t size) {
    size = getAlignedSize(size);
    size_t retAddr = this->weightPeak;
    this->weightPeak += size;
    return retAddr;
}

size_t MemoryAllocator::allocIO(size_t size) {
    size = getAlignedSize(size);
    size_t retAddr = this->ioPeak;
    this->ioPeak += size;
    return retAddr;
}

size_t MemoryAllocator::heapAlloc(size_t size) {
    size = getAlignedSize(size);
    this->heapPeak += size;
    // 断言：确保 heap 不会与 weight, IO, other 区域发生碰撞
    IT_ASSERT(this->memPoolSize >=
                  this->weightPeak + this->ioPeak + this->peak + this->heapPeak,
              "Heap allocation exceeds memory pool capacity.");
    size_t retAddr = this->memPoolSize - this->heapPeak;
    return retAddr;
}

void MemoryAllocator::freeHeap() { this->heapPeak = 0; }

void *MemoryAllocator::getWeightPtr() {
    IT_ASSERT(hasMemPool, "Memory pool not allocated.");
    return this->memPoolPtr;
}

void *MemoryAllocator::getIOPtr() {
    IT_ASSERT(hasMemPool, "Memory pool not allocated.");
    return static_cast<uint8_t *>(this->memPoolPtr) + this->weightPeak;
}

void *MemoryAllocator::getPtr() {
    IT_ASSERT(hasMemPool, "Memory pool not allocated.");
    return static_cast<uint8_t *>(this->memPoolPtr) + this->weightPeak +
           this->ioPeak;
}

void *MemoryAllocator::getHeapPtr() {
    IT_ASSERT(hasMemPool, "Memory pool not allocated.");
    return this->memPoolPtr;
}

size_t MemoryAllocator::getAlignedSize(size_t size) {
    if (size == 0)
        return 0;
    return ((size - 1) / this->alignment + 1) * this->alignment;
}

void MemoryAllocator::info() const {
    std::cout << "MemoryAllocator Info:\n"
              << "  - MemPool Status: "
              << (hasMemPool ? "Allocated" : "Not Allocated") << "\n"
              << "  - MemPool Size: " << memPoolSize << " bytes\n"
              << "  - Weight Peak: " << weightPeak << " bytes\n"
              << "  - IO Peak: " << ioPeak << " bytes\n"
              << "  - Other Peak: " << peak << " bytes\n"
              << "  - Heap Peak: " << heapPeak << " bytes\n";
}

} // namespace infini
