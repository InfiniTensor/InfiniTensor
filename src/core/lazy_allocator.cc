#include "core/lazy_allocator.h"
#include <utility>

namespace infini {
    LazyAllocator::LazyAllocator(Runtime runtime, size_t alignment) : 
                            runtime(runtime), alignment(alignment), used(0), peak(0) {}

    LazyAllocator::~LazyAllocator() {}

    size_t LazyAllocator::alloc(size_t size) {
        // 这里保守考虑，使用 size + alignment 作为要寻找的空闲块大小
            auto it = this->freeBlocks.lower_bound(freeBlockInfo{.addr = (size_t)0, 
                                                           .blockSize = size + this->alignment});

        size_t retAddr = this->peak;
        if (it != this->freeBlocks.end()) {
            // 找到了可以分配的空内存块
            size_t blockSize = it->blockSize;
            retAddr = it->addr;
            size_t tailAddr = getAlignedTailAddr(retAddr + size);
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
            this->peak = getAlignedTailAddr(this->peak + size);
            this->used += this->peak - retAddr;
        }
        return retAddr;
    }

    void LazyAllocator::free(size_t addr, size_t size) {
        auto tailAddr = getAlignedTailAddr(addr + size);
        size_t currBlockSize = tailAddr - addr;
        freeBlockInfo block = {.addr = addr, 
                               .blockSize = currBlockSize};
        this->headAddrToBlockSize[addr] = currBlockSize;
        this->tailAddrToBlockSize[tailAddr] = currBlockSize;
        auto preFreeBlockIter = this->tailAddrToBlockSize.find(addr);
        auto subFreeBlockIter = this->headAddrToBlockSize.find(tailAddr);
        if (preFreeBlockIter != this->tailAddrToBlockSize.end()) {
            // 需要释放的内存块的头地址是某个空闲块的尾，将二者进行合并
            size_t preBlockSize = preFreeBlockIter->second;
            this->headAddrToBlockSize.erase(addr);
            this->headAddrToBlockSize[addr - preBlockSize] += currBlockSize;
            this->tailAddrToBlockSize.erase(addr);
            this->tailAddrToBlockSize[tailAddr] += preBlockSize;
            addr -= preBlockSize;
            block.addr = addr;
            block.blockSize += preBlockSize;
            // 删掉原来的前相邻空闲块，这里是否需要先 find 拿到迭代器，再进行删除？（以防之后修改代码出问题
            this->freeBlocks.erase(freeBlockInfo{.addr = addr, 
                                           .blockSize=preBlockSize});
        }
        if (subFreeBlockIter != this->headAddrToBlockSize.end()) {
            // 需要释放的内存块的尾地址是某个空闲块的头，将二者进行合并
            auto subBlockSize = subFreeBlockIter->second;
            this->headAddrToBlockSize.erase(tailAddr);
            this->headAddrToBlockSize[addr] += subBlockSize;
            this->tailAddrToBlockSize.erase(tailAddr);
            this->tailAddrToBlockSize[tailAddr + subBlockSize] += currBlockSize;
            tailAddr += subBlockSize;
            block.blockSize += subBlockSize;
            // 删掉原来的后相邻内存块
            this->freeBlocks.erase(freeBlockInfo{.addr = tailAddr - subBlockSize, 
                                           .blockSize = subBlockSize});
        }
        this->freeBlocks.insert(block);
    }

    void* LazyAllocator::ptr() {
        return runtime->alloc(this->peak);
    }

    size_t LazyAllocator::getAlignedTailAddr(size_t baseAddr) {
        return ((baseAddr - 1) / this->alignment + 1) * this->alignment;
    }

    void LazyAllocator::info() {
        std::cout << "Used memory: " << this->used << ", peak memory: " << this->peak << std::endl;
    }

} // namespace infini
