#pragma once
#include <cstddef>
#include <map>
#include <unordered_set>
#include "core/runtime.h"

namespace infini {

class LazyAllocator {
    protected:
        Runtime runtime;

        size_t used;

        size_t peak;

        size_t alignment;

        // pointer to the memory actually allocated
        void *ptr;

        struct freeBlockInfo {
            size_t addr;
            size_t blockSize;
        };

        struct cmpFreeBlockInfo {
            bool operator()(const freeBlockInfo &a, const freeBlockInfo &b) const {
                return (a.blockSize != b.blockSize) ? (a.blockSize < b.blockSize)
                                                    : (a.addr < b.addr);
            }
        };

        // free balanced tree, maintains all free memory blocks
        std::set<freeBlockInfo, cmpFreeBlockInfo> freeBlocks;

        // key: head address offset of the free memory block
        // value: blockSize of the block
        std::unordered_map<size_t, size_t> headAddrToBlockSize;
        
        // key: tail address offset of the free memory block
        // value: blockSize of the block
        std::unordered_map<size_t, size_t> tailAddrToBlockSize;       

    public:
        LazyAllocator(Runtime runtime, size_t alignment);
        
        virtual ~LazyAllocator();

        // function: simulate memory allocation
        // arguments：
        //     size: size of memory block to be allocated
        // return: head address offset of the allocated memory block
        size_t alloc(size_t size);

        // function: simulate memory free
        // arguments:
        //     addr: head address offset of memory block to be free
        //     size: size of memory block to be freed
        void free(size_t addr, size_t size);

        // function: perform actual memory allocation
        // return: pointer to the head address of the allocated memory
        void* getPtr();

        // function: memory alignment, rouned up
        // return: size of the aligned memory block
        size_t getAlignedSize(size_t size);

        void info();
};

} // namespace infini
