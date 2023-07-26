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
            auto it = freeBlocks.lower_bound(freeBlockInfo{.addr = (size_t)0, 
                                                           .blockSize = size + alignment});

        size_t retAddr = peak;
        if (it != freeBlocks.end()) {
            // 找到了可以分配的空内存块
            size_t blockSize = it->blockSize;
            retAddr = it->addr;
            size_t tailAddr = getAlignedTailAddr(retAddr + size);
            // 更新空闲块地址集合
            headAddrToBlockSize.erase(retAddr);
            tailAddrToBlockSize.erase(tailAddr);
            // 内存块分裂
            if (blockSize > tailAddr - retAddr) {
                freeBlockInfo newBlock = {.addr = tailAddr, 
                                          .blockSize = blockSize - (tailAddr - retAddr)};
                headAddrToBlockSize[tailAddr] = newBlock.blockSize;
                tailAddrToBlockSize[retAddr + blockSize] = newBlock.blockSize;
                freeBlocks.insert(newBlock);
            }
            // 更新空闲平衡树
            freeBlocks.erase(it);
            used += tailAddr - retAddr;
        } else {
            // 已分配的内存空间大小不足以进行再分配，需要扩充
            retAddr = peak;
            peak = getAlignedTailAddr(peak + size);
            used += peak - retAddr;
        }
        return retAddr;
    }

    void LazyAllocator::free(size_t addr, size_t size) {
        auto tailAddr = getAlignedTailAddr(addr + size);
        size_t currBlockSize = tailAddr - addr;
        freeBlockInfo block = {.addr = addr, 
                               .blockSize = currBlockSize};
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
            // 删掉原来的前相邻空闲块，这里是否需要先 find 拿到迭代器，再进行删除？（以防之后修改代码出问题
            freeBlocks.erase(freeBlockInfo{.addr = addr, 
                                           .blockSize=preBlockSize});
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
            // 删掉原来的后相邻内存块
            freeBlocks.erase(freeBlockInfo{.addr = tailAddr - subBlockSize, 
                                           .blockSize = subBlockSize});
        }
        freeBlocks.insert(block);
    }

    Blob LazyAllocator::ptr() {
        return runtime->allocBlob(peak);
    }

    size_t LazyAllocator::getAlignedTailAddr(size_t baseAddr) {
        if (alignment == 0) {
            return baseAddr;
        }
        return ((baseAddr - 1) / alignment + 1) * alignment;
    }

    void LazyAllocator::info() {
        std::cout << "Used memory: " << used << ", peak memory: " << peak << std::endl;
    }

} // namespace infini
