#include "core/lazy_allocator.h"
#include <algorithm>
#include <limits>
#include <utility>

namespace infini {

constexpr size_t alignmentInBytesForInfiniRuntime = 256;

static size_t checkedAdd(size_t lhs, size_t rhs, const char *message) {
    IT_ASSERT(lhs <= std::numeric_limits<size_t>::max() - rhs, message);
    return lhs + rhs;
}

LazyAllocator::LazyAllocator(Runtime runtime) : runtime(runtime) {
    if (runtime->isInfini()) {
        alignment = alignmentInBytesForInfiniRuntime;
    } else {
        // Native CPU allocations only need alignment for the widest scalar
        // type supported by Tensor.
        alignment = sizeof(uint64_t);
    }
}

LazyAllocator::~LazyAllocator() = default;

void LazyAllocator::init() {
    used = 0;
    peak = 0;
    freeBlocks.clear();
    headAddrToBlockSize.clear();
    tailAddrToBlockSize.clear();
}

void LazyAllocator::reset() {
    init();
    weightPeak = 0;
    heapPeak = 0;
    hasMemPool = false;
    memPoolSize = 0;
    ptr.reset();
    weightPtr.reset();
    memPoolPtr.reset();
    heapBlobs.clear();
}

void LazyAllocator::resetWeightPlan() {
    weightPeak = 0;
    weightPtr.reset();
}

void LazyAllocator::setMemPool(size_t memPoolSize) {
    IT_ASSERT(memPoolSize > 0);
    if (!this->hasMemPool) {
        auto pool = runtime->allocBlob(memPoolSize);
        this->hasMemPool = true;
        this->memPoolSize = memPoolSize;
        this->memPoolPtr = std::move(pool);
    }
}

bool LazyAllocator::getMemPoolStatus() { return this->hasMemPool; }

size_t LazyAllocator::alloc(size_t size) {
    // pad the size to the multiple of alignment
    size = this->getAlignedSize(size);
    if (size == 0)
        return peak;
    auto it = this->freeBlocks.lower_bound(freeBlockInfo{(size_t)0, size});

    size_t retAddr = this->peak;
    if (it != this->freeBlocks.end()) {
        // found an alvailable free memory block for allocation
        size_t blockSize = it->blockSize;
        retAddr = it->addr;
        size_t tailAddr = checkedAdd(retAddr, size, "Memory offset overflow");
        // update the map of head and tail address offset of memory blocks
        this->headAddrToBlockSize.erase(retAddr);
        this->tailAddrToBlockSize.erase(tailAddr);
        // memory block splitting
        if (blockSize > tailAddr - retAddr) {
            freeBlockInfo newBlock = {tailAddr,
                                      blockSize - (tailAddr - retAddr)};
            this->headAddrToBlockSize[tailAddr] = newBlock.blockSize;
            this->tailAddrToBlockSize[retAddr + blockSize] = newBlock.blockSize;
            this->freeBlocks.insert(newBlock);
        }
        // update the free balanced tree
        this->freeBlocks.erase(it);
        this->used =
            checkedAdd(this->used, tailAddr - retAddr, "Used memory overflow");
    } else {
        // the allocated memory space is not sufficient for reallocation, it
        // needs to be extended
        auto blockTailWithPeak = this->tailAddrToBlockSize.find(this->peak);
        if (blockTailWithPeak != this->tailAddrToBlockSize.end()) {
            // there is a free block located at the end of the currently
            // allocated memory, where this free block has its tail address as
            // 'peak'
            retAddr = this->peak - blockTailWithPeak->second;
            IT_ASSERT(blockTailWithPeak->second < size);
            this->peak =
                checkedAdd(this->peak, size - blockTailWithPeak->second,
                           "Memory peak overflow");
            // updata freeBlocks, headAddrToBlockSize and tailAddrToBlockSize
            freeBlockInfo endBlock = {retAddr, blockTailWithPeak->second};
            this->freeBlocks.erase(endBlock);
            this->headAddrToBlockSize.erase(endBlock.addr);
            this->tailAddrToBlockSize.erase(endBlock.addr + endBlock.blockSize);
        } else {
            this->peak = checkedAdd(this->peak, size, "Memory peak overflow");
        }
        this->used = checkedAdd(this->used, size, "Used memory overflow");
    }

    return retAddr;
}

size_t LazyAllocator::allocWeight(size_t size) {
    IT_ASSERT(this->weightPtr == nullptr);
    size = this->getAlignedSize(size);
    size_t retAddr = this->weightPeak;
    this->weightPeak =
        checkedAdd(this->weightPeak, size, "Weight memory overflow");
    return retAddr;
}

size_t LazyAllocator::heapAlloc(size_t size) {
    size = this->getAlignedSize(size);
    const auto newHeapPeak =
        checkedAdd(this->heapPeak, size, "Heap memory overflow");
    const auto graphPeak =
        checkedAdd(this->weightPeak, this->peak, "Graph memory overflow");
    const auto totalPeak =
        checkedAdd(graphPeak, newHeapPeak, "Total memory overflow");
    IT_ASSERT(this->memPoolSize >= totalPeak,
              "Fixed memory pool capacity is insufficient for heap data");
    this->heapPeak = newHeapPeak;
    return this->memPoolSize - this->heapPeak;
}

void LazyAllocator::rollbackHeap(size_t previousPeak) {
    IT_ASSERT(previousPeak <= heapPeak, "Invalid heap rollback position");
    heapPeak = previousPeak;
}

bool LazyAllocator::hasLiveHeapBlobs() {
    heapBlobs.erase(
        std::remove_if(heapBlobs.begin(), heapBlobs.end(),
                       [](const auto &blob) { return blob.expired(); }),
        heapBlobs.end());
    return !heapBlobs.empty();
}

void LazyAllocator::freeHeap() {
    IT_ASSERT(!hasLiveHeapBlobs(),
              "Cannot free heap while heap tensors are still alive");
    this->heapPeak = 0;
}

void LazyAllocator::free(size_t addr, size_t size) {
    size = getAlignedSize(size);
    if (size == 0)
        return;
    auto tailAddr = checkedAdd(addr, size, "Memory offset overflow");
    freeBlockInfo block = {addr, tailAddr - addr};
    this->headAddrToBlockSize[addr] = block.blockSize;
    this->tailAddrToBlockSize[tailAddr] = block.blockSize;
    auto preFreeBlockIter = this->tailAddrToBlockSize.find(addr);
    auto subFreeBlockIter = this->headAddrToBlockSize.find(tailAddr);
    if (preFreeBlockIter != this->tailAddrToBlockSize.end()) {
        // the head address of the memory block to be freed matches the end of a
        // free block, merge them together
        size_t preBlockSize = preFreeBlockIter->second;
        this->headAddrToBlockSize.erase(block.addr);
        this->headAddrToBlockSize[block.addr - preBlockSize] += block.blockSize;
        this->tailAddrToBlockSize.erase(block.addr);
        this->tailAddrToBlockSize[tailAddr] += preBlockSize;
        block.addr -= preBlockSize;
        block.blockSize += preBlockSize;
        // delete the preceding adjacent free block
        this->freeBlocks.erase(freeBlockInfo{block.addr, preBlockSize});
    }
    if (subFreeBlockIter != this->headAddrToBlockSize.end()) {
        // the tail address of the memory block to be freed matches the start of
        // a free block, merge them together
        auto subBlockSize = subFreeBlockIter->second;
        this->headAddrToBlockSize.erase(tailAddr);
        this->headAddrToBlockSize[block.addr] += subBlockSize;
        this->tailAddrToBlockSize.erase(tailAddr);
        this->tailAddrToBlockSize[tailAddr + subBlockSize] += block.blockSize;
        tailAddr += subBlockSize;
        block.blockSize += subBlockSize;
        // delete the succeeding adjacent memory block
        this->freeBlocks.erase(
            freeBlockInfo{tailAddr - subBlockSize, subBlockSize});
    }
    this->freeBlocks.insert(block);
    IT_ASSERT(this->used >= size, "Freed memory exceeds used memory");
    this->used -= size;
}

void *LazyAllocator::getPtr() {
    auto storage = prepareActivationStorage();
    commitActivationStorage(storage);
    if (hasMemPool)
        return static_cast<uint8_t *>(storage->getPtr<void *>()) + weightPeak;
    return storage->getPtr<void *>();
}

Blob LazyAllocator::prepareActivationStorage(bool exactCapacity,
                                             bool forceNewStorage) {
    if (hasMemPool) {
        const auto graphPeak =
            checkedAdd(weightPeak, peak, "Graph memory overflow");
        const auto totalPeak =
            checkedAdd(graphPeak, heapPeak, "Total memory overflow");
        IT_ASSERT(memPoolSize >= totalPeak,
                  "Fixed memory pool capacity is insufficient");
        return memPoolPtr;
    }

    const auto currentCapacity = ptr ? ptr->getBytes() : 0;
    if (!forceNewStorage && ptr &&
        ((!exactCapacity && peak <= currentCapacity) ||
         (exactCapacity && peak == currentCapacity)))
        return ptr;

    size_t newCapacity = peak;
    if (forceNewStorage) {
        newCapacity = std::max(newCapacity, currentCapacity);
    } else if (!exactCapacity && currentCapacity > 0) {
        const auto grownCapacity =
            checkedAdd(currentCapacity, currentCapacity / 2,
                       "Activation pool capacity overflow");
        newCapacity = std::max(newCapacity, grownCapacity);
    }
    return runtime->allocBlob(newCapacity);
}

void LazyAllocator::commitActivationStorage(const Blob &storage) {
    IT_ASSERT(storage != nullptr, "Cannot commit null activation storage");
    if (hasMemPool) {
        IT_ASSERT(storage == memPoolPtr,
                  "Fixed memory pool storage cannot be replaced");
    } else {
        ptr = storage;
    }
}

bool LazyAllocator::isCurrentActivationStorage(const Blob &storage) const {
    return storage && storage == (hasMemPool ? memPoolPtr : ptr);
}

Blob LazyAllocator::getActivationBlob(size_t offset, size_t bytes) {
    auto storage = prepareActivationStorage();
    commitActivationStorage(storage);
    return getActivationBlob(storage, offset, bytes);
}

Blob LazyAllocator::getActivationBlob(const Blob &storage, size_t offset,
                                      size_t bytes) const {
    IT_ASSERT(storage != nullptr, "Activation storage is not allocated");
    if (!hasMemPool) {
        return make_ref<BlobObj>(storage, offset, bytes);
    }
    IT_ASSERT(weightPeak <= memPoolSize);
    IT_ASSERT(offset <= memPoolSize - weightPeak);
    return make_ref<BlobObj>(
        storage, checkedAdd(weightPeak, offset, "Memory offset overflow"),
        bytes);
}

void *LazyAllocator::getWeightPtr() {
    if (!hasMemPool) {
        if (this->weightPtr == nullptr) {
            this->weightPtr = runtime->allocBlob(this->weightPeak);
            // #ifdef DEBUG_MODE
            //         printf("LazyAllocator really alloc weight: %p %lu
            //         bytes\n",
            //                this->weightPtr, weightPeak);
            // #endif
        }
        return this->weightPtr->getPtr<void *>();
    } else {
        return this->memPoolPtr->getPtr<void *>();
    }
}

Blob LazyAllocator::getWeightBlob(size_t offset, size_t bytes) {
    getWeightPtr();
    const auto &storage = hasMemPool ? memPoolPtr : weightPtr;
    return make_ref<BlobObj>(storage, offset, bytes);
}

void *LazyAllocator::getHeapPtr() {
    IT_ASSERT(hasMemPool);
    return this->memPoolPtr->getPtr<void *>();
}

Blob LazyAllocator::getHeapBlob(size_t offset, size_t bytes) {
    getHeapPtr();
    auto blob = make_ref<BlobObj>(memPoolPtr, offset, bytes);
    heapBlobs.emplace_back(blob);
    return blob;
}

size_t LazyAllocator::getAlignedSize(size_t size) {
    if (size == 0)
        return 0;
    IT_ASSERT(size <= std::numeric_limits<size_t>::max() - alignment + 1,
              "Memory alignment overflow");
    return ((size - 1) / this->alignment + 1) * this->alignment;
}

void LazyAllocator::info() {
    std::cout << "Used memory: " << this->used + this->weightPeak
              << ", peak memory: " << this->peak + this->weightPeak
              << std::endl;
}

} // namespace infini
