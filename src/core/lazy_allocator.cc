#include "core/lazy_allocator.h"
#include <utility>

namespace infini {
    LazyAllocator::LazyAllocator(Runtime runtime, size_t alignment) : 
                            runtime(runtime), alignment(alignment) {
        used = 0;
        peak = 0;
        ptr = nullptr;
    }

    LazyAllocator::~LazyAllocator() { runtime->dealloc(this->ptr); }

    size_t LazyAllocator::alloc(size_t size) {
        IT_ASSERT(this->ptr == nullptr);
        // 将 size 填充至 alignment 的倍数
        size = this->getAlignedSize(size);
        // 这里保守考虑，使用 size + alignment 作为要寻找的空闲块大小
        auto it = this->freeBlocks.lower_bound(freeBlockInfo{.addr = (size_t)0, 
                                                             .blockSize = size});

        size_t retAddr = this->peak;
        if (it != this->freeBlocks.end()) {
            // 找到了可以分配的空内存块
            size_t blockSize = it->blockSize;
            retAddr = it->addr;
            size_t tailAddr = retAddr + size;
            // 更新空闲块地址集合
            this->headAddrToBlockSize.erase(retAddr);
            this->tailAddrToBlockSize.erase(tailAddr);
            // 内存块分裂
            if (blockSize > tailAddr - retAddr) {
                freeBlockInfo newBlock = {.addr = tailAddr, 
                                          .blockSize = blockSize - (tailAddr - retAddr)};
                this->headAddrToBlockSize[tailAddr] = newBlock.blockSize;
                this->tailAddrToBlockSize[retAddr + blockSize] = newBlock.blockSize;
                this->freeBlocks.insert(newBlock);
            }
            // 更新空闲平衡树
            this->freeBlocks.erase(it);
            this->used += tailAddr - retAddr;
        } else {
            // 已分配的内存空间大小不足以进行再分配，需要扩充
            retAddr = this->peak;
            this->peak = this->peak + size;
            this->used += this->peak - retAddr;
        }
        // printf("LazyAllocator alloc: %lu %lu bytes\n", retAddr, size);

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
            // 需要释放的内存块的头地址是某个空闲块的尾，将二者进行合并
            size_t preBlockSize = preFreeBlockIter->second;
            this->headAddrToBlockSize.erase(block.addr);
            this->headAddrToBlockSize[block.addr - preBlockSize] += block.blockSize;
            this->tailAddrToBlockSize.erase(block.addr);
            this->tailAddrToBlockSize[tailAddr] += preBlockSize;
            block.addr -= preBlockSize;
            block.blockSize += preBlockSize;
            // 删掉原来的前相邻空闲块，这里是否需要先 find 拿到迭代器，再进行删除？（以防之后修改代码出问题
            this->freeBlocks.erase(freeBlockInfo{block.addr, preBlockSize});
        }
        if (subFreeBlockIter != this->headAddrToBlockSize.end()) {
            // 需要释放的内存块的尾地址是某个空闲块的头，将二者进行合并
            auto subBlockSize = subFreeBlockIter->second;
            this->headAddrToBlockSize.erase(tailAddr);
            this->headAddrToBlockSize[block.addr] += subBlockSize;
            this->tailAddrToBlockSize.erase(tailAddr);
            this->tailAddrToBlockSize[tailAddr + subBlockSize] += block.blockSize;
            tailAddr += subBlockSize;
            block.blockSize += subBlockSize;
            // 删掉原来的后相邻内存块
            this->freeBlocks.erase(freeBlockInfo{tailAddr - subBlockSize, subBlockSize});
        }
        this->freeBlocks.insert(block);
        this->used -= size;
        // printf("LazyAllocator free: %lu %lu bytes\n", addr, size);
    }

    void* LazyAllocator::getPtr() {
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
        std::cout << "Used memory: " << this->used << ", peak memory: " << this->peak << std::endl;
    }

} // namespace infini
