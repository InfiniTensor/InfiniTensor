#pragma once
#include "core/runtime.h"
#include "core/tensor.h"
#ifdef BUILD_TEST
#include "gtest/gtest.h"
#endif
#include <cstddef>
#include <map>
#include <unordered_set>

namespace infini {

class LazyAllocator {
  private:
#ifdef BUILD_TEST
    FRIEND_TEST(LazyAllocator, testMergeFreeBlocks);

    FRIEND_TEST(LazyAllocator, testAllocWithEndFreeBlock);
#endif

    Runtime runtime;

    size_t used = 0;

    size_t peak = 0;

    size_t persistentPeak = 0;

    size_t alignment;

    // pointer to the memory actually allocated
    void *ptr = nullptr;

    // pointer to the persistent memory space
    void *persistentPtr = nullptr;

    // // a cache designed for a batch size that has already occurred
    // std::unordered_map<size_t, std::unordered_map<TensorObj *, size_t>>
    // batchsizeToTensorOffset;

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
    LazyAllocator(Runtime runtime);

    virtual ~LazyAllocator();

    void init();

    // function: simulate memory allocation
    // arguments：
    //     size: size of memory block to be allocated
    // return: head address offset of the allocated memory block
    size_t alloc(size_t size);

    size_t allocPersistent(size_t size);

    // function: simulate memory free
    // arguments:
    //     addr: head address offset of memory block to be free
    //     size: size of memory block to be freed
    void free(size_t addr, size_t size);

    // function: perform actual memory allocation
    // return: pointer to the head address of the allocated memory
    void *getPtr();

    // void addCache(size_t batchsize, std::unordered_map<TensorObj *, size_t>);

    // std::unordered_map<TensorObj *, size_t> getCache(size_t batchsize);

    void *getPersistentPtr();

    void info();

  private:
    // function: memory alignment, rouned up
    // return: size of the aligned memory block
    size_t getAlignedSize(size_t size);
};

} // namespace infini
