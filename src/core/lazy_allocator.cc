#include "core/lazy_allocator.h"
#include <utility>

namespace infini {
    LazyAllocator::LazyAllocator(Runtime runtime, size_t alignment) : runtime(runtime), alignment(alignment) {}

    LazyAllocator::~LazyAllocator() {}

    void LazyAllocator::init(size_t size) {
        peak = size;
        used = size;
    }

    size_t LazyAllocator::alloc(size_t size, size_t alignment) {
        // 这里保守考虑，使用 size + alignment 作为要寻找的空闲块大小
        auto it = freeBlocks.lower_bound(freeBlockInfo((size_t)0, size + alignment));

        size_t retAddr = peak;
        if (it != freeBlocks.end()) {
            // 找到了可以分配的空内存块
            size_t blockSize = it->blockSize;
            retAddr = it->addr;
            size_t tailAddr = getAlignedTailAddr(retAddr);
            // 更新空闲块地址集合
            headAddrToBlockSize.erase(retAddr);
            tailAddrToBlockSize.erase(tailAddr);
            // 内存块分裂
            if (blockSize > tailAddr - retAddr) {
                freeBlockInfo newBlock = freeBlockInfo(tailAddr, blockSize - (tailAddr - retAddr));
                headAddrToBlockSize[tailAddr] = newBlock.blockSize;
                tailAddrToBlockSize[retAddr + blockSize] = newBlock.blockSize;
                freeBlocks.insert(newBlock);
            }
            // 更新空闲平衡树
            freeBlocks.erase(it);
        } else {
            // 已分配的内存空间大小不足以进行再分配，需要扩充
            retAddr = peak;
            peak = getAlignedTailAddr(peak + size);
        }
        return retAddr;
    }

    void LazyAllocator::free(size_t addr, size_t size) {
        auto tailAddr = getAlignedTailAddr(addr + size);
        size_t currBlockSize = tailAddr - addr;
        freeBlockInfo block = freeBlockInfo(addr, tailAddr - addr);
        headAddrToBlockSize[addr] = currBlockSize;
        tailAddrToBlockSize[tailAddr] = currBlockSize;
        auto preFreeBlockIter = tailAddrToBlockSize.find(addr);
        auto subFreeBlockIter = headAddrToBlockSize.find(tailAddr);
        if (preFreeBlockIter != tailAddrToBlockSize.end()) {
            // 需要释放的内存块的头地址是某个空闲块的尾，将二者进行合并
            size_t preBlockSize = preFreeBlockIter->second;
            headAddrToBlockSize.erase(addr);
            headAddrToBlockSize[addr - preBlockSize] += currBlockSize;
            tailAddrToBlockSize.erase(addr);
            tailAddrToBlockSize[tailAddr] += preBlockSize;
            addr -= preBlockSize;
            block.addr = addr;
            block.blockSize += preBlockSize;
        }
        if (subFreeBlockIter != headAddrToBlockSize.end()) {
            // 需要释放的内存块的尾地址是某个空闲块的头，将二者进行合并
            auto subBlockSize = subFreeBlockIter->second;
            headAddrToBlockSize.erase(tailAddr);
            headAddrToBlockSize[addr] += subBlockSize;
            tailAddrToBlockSize.erase(tailAddr);
            tailAddrToBlockSize[tailAddr + subBlockSize] += currBlockSize;
            tailAddr += subBlockSize;
            block.blockSize += subBlockSize;
        }
        freeBlocks.insert(block);
    }

    Blob LazyAllocator::ptr() {
        return runtime->allocBlob(peak);
    }

    size_t LazyAllocator::getAlignedTailAddr(size_t baseAddr) {
        return ((baseAddr - 1) / alignment + 1) * alignment;
    }

} // namespace infini