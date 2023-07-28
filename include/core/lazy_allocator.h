#pragma once
#include <cstddef>
#include <map>
#include <unordered_set>
#include "core/runtime.h"

namespace infini {

class LazyAllocator {
    protected:
        // 用于调用 runtime->allocBlob() 函数进行实际的内存分配
        Runtime runtime;

        // 记录当前使用的内存大小
        size_t used;

        // 记录推理过程中的内存峰值
        size_t peak;

        // 内存对齐
        size_t alignment;

        // 指向实际分配的内存的指针
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

        // 空闲平衡树，维护所有空闲内存块
        // key: 空闲内存块大小，val: 内存块起始地址偏移量
        std::set<freeBlockInfo, cmpFreeBlockInfo> freeBlocks;
        // map<size_t, size_t> freeSizeToHeadAddr;

        // 空闲块起始地址集合，维护所有空闲内存块的起始地址偏移量，用于碎片回收
        std::unordered_map<size_t, size_t> headAddrToBlockSize;
        
        // 空闲块结尾地址集合，维护所有空闲内存块的结尾地址偏移量，用于碎片回收
        std::unordered_map<size_t, size_t> tailAddrToBlockSize;       

    public:
        LazyAllocator(Runtime runtime, size_t alignment);
        
        virtual ~LazyAllocator();

        // 功能：模拟分配内存。   
        // 输入参数：
        //     size：需要分配的内存大小。
        //     alignment：内存对齐量。
        // 返回值：所分配内存块的起始地址偏移量。 
        size_t alloc(size_t size);

        // 功能：模拟释放内存。
        // 输入参数：
        //     addr：需要释放的内存起始地址偏移量。
        //     size：需要释放的 Tensor 大小
        void free(size_t addr, size_t size);

        // 功能：进行实际的内存分配
        // 返回值：指向所分配内存起始地址的指针
        void* getPtr();

        // 功能：内存对齐，向上取整
        // 返回值：对齐后的尾地址
        size_t getAlignedSize(size_t size);

        void info();
};

} // namespace infini
