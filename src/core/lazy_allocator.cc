#include "core/lazy_allocator.h"
#include <utility>

namespace infini {
LazyAllocator::LazyAllocator(Runtime runtime) : runtime(runtime) {
    used = 0;
    peak = 0;
    ptr = nullptr;
    // 'alignment' defaults to sizeof(uint64_t), because it is the length of the
    // longest data type currently supported by the DataType field of the tensor
    alignment = sizeof(uint64_t);
}

LazyAllocator::~LazyAllocator() {
    if (this->ptr != nullptr) {
        runtime->dealloc(this->ptr);
    }
}

size_t LazyAllocator::alloc(size_t size) {
    IT_ASSERT(this->ptr == nullptr);
    // pad the size to the multiple of alignment
    size = this->getAlignedSize(size);
    auto it = this->freeBlocks.lower_bound(
        freeBlockInfo{.addr = (size_t)0, .blockSize = size});

    size_t retAddr = this->peak;
    if (it != this->freeBlocks.end()) {
        // found an alvailable free memory block for allocation
        size_t blockSize = it->blockSize;
        retAddr = it->addr;
        size_t tailAddr = retAddr + size;
        // update the map of head and tail address offset of memory blocks
        this->headAddrToBlockSize.erase(retAddr);
        this->tailAddrToBlockSize.erase(tailAddr);
        // memory block splitting
        if (blockSize > tailAddr - retAddr) {
            freeBlockInfo newBlock = {.addr = tailAddr,
                                      .blockSize =
                                          blockSize - (tailAddr - retAddr)};
            this->headAddrToBlockSize[tailAddr] = newBlock.blockSize;
            this->tailAddrToBlockSize[retAddr + blockSize] = newBlock.blockSize;
            this->freeBlocks.insert(newBlock);
        }
        // update the free balanced tree
        this->freeBlocks.erase(it);
        this->used += tailAddr - retAddr;
    } else {
        // the allocated memory space is not sufficient for reallocation, it
        // needs to be extended
        this->peak = this->peak + size;
        this->used += size;
    }

    return retAddr;
}

void LazyAllocator::free(size_t addr, size_t size) {
    IT_ASSERT(this->ptr == nullptr);
    size = getAlignedSize(size);
    auto tailAddr = addr + size;
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
    this->used -= size;
}

void *LazyAllocator::getPtr() {
    if (this->ptr == nullptr) {
        this->ptr = runtime->alloc(this->peak);
        printf("LazyAllocator really alloc: %p %lu bytes\n", this->ptr, peak);
    }
    return this->ptr;
}

size_t LazyAllocator::getAlignedSize(size_t size) {
    return ((size - 1) / this->alignment + 1) * this->alignment;
}

void LazyAllocator::info() {
    std::cout << "Used memory: " << this->used
              << ", peak memory: " << this->peak << std::endl;
}

} // namespace infini
