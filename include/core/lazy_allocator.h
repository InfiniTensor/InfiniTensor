#pragma once
#include <cstddef>
#include <map>
#include <unordered_set>
#include "core/runtime.h"

namespace infini {

class LazyAllocator {
    protected:
        Runtime runtime;

        // 记录推理过程中的内存峰值
        size_t peak;

        // 空闲平衡树，维护所有空闲内存块
        // key: 空闲内存块大小，val: 内存块起始地址偏移量
        std::map<size_t, size_t> freeSizeToHeadAddr;

        // 空闲块起始地址集合，维护所有空闲内存块的起始地址偏移量，用于碎片回收
        std::unordered_set<size_t> freeBlocksHeadAddr;
        
        // 空闲块结尾地址集合，维护所有空闲内存块的结尾地址偏移量，用于碎片回收
        std::unordered_set<size_t> freeBlocksTailAddr; 
    public:
        LazyAllocator(Runtime runtime);
        
        virtual ~LazyAllocator();

        // 功能：模拟分配内存。   
        // 输入参数：
        //     size：需要分配的内存大小。
        //     alignment：内存对齐量。
        // 返回值：所分配内存块的起始地址偏移量。 
        void init(size_t size);

        size_t alloc(size_t size, size_t alignment);

        // 功能：模拟释放内存。
        // 输入参数：
        //     addr：需要释放的内存起始地址偏移量。
        void free(size_t addr);

        // 功能：进行实际的内存分配
        // 返回值：指向所分配内存起始地址的指针
        Blob ptr();
};

} // namespace infini