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

    size_t weightPeak = 0;

    size_t heapPeak = 0;

    // How many times activation storage has actually been asked of the
    // runtime. Storage that gets reused does not count, which is the whole
    // point: this is the number a workload that keeps changing shape is
    // trying to hold down. It is a tally over the allocator's life and so
    // outlives `init` and `reset` -- a layout that was rolled back still
    // allocated, and a measurement that forgot it would flatter the result.
    size_t activationAllocations = 0;

    size_t alignment;

    bool hasMemPool = false;

    size_t memPoolSize = 0;

    // pointer to the memory actually allocated
    Blob ptr;

    // pointer to the weight memory space
    Blob weightPtr;

    // memory pool ptr
    Blob memPoolPtr;

    // Weak references prevent heap offsets from being reused while a cloned
    // tensor still points into the fixed memory pool.
    vector<WRef<BlobObj>> heapBlobs;

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

    void reset();

    void resetWeightPlan();

    void setMemPool(size_t memPoolSize);

    bool getMemPoolStatus();

    // function: simulate memory allocation
    // arguments：
    //     size: size of memory block to be allocated
    // return: head address offset of the allocated memory block
    size_t alloc(size_t size);

    size_t allocWeight(size_t size);

    size_t heapAlloc(size_t size);

    void rollbackHeap(size_t previousPeak);

    void freeHeap();

    bool hasLiveHeapBlobs();

    // function: simulate memory free
    // arguments:
    //     addr: head address offset of memory block to be free
    //     size: size of memory block to be freed
    void free(size_t addr, size_t size);

    // function: perform actual memory allocation
    // return: pointer to the head address of the allocated memory
    void *getPtr();

    Blob prepareActivationStorage(bool exactCapacity = false,
                                  bool forceNewStorage = false);

    void commitActivationStorage(const Blob &storage);

    bool isCurrentActivationStorage(const Blob &storage) const;

    size_t getHeapPeak() const { return heapPeak; }

    size_t getWeightPeak() const { return weightPeak; }

    // What the activations need, as against what is being held for them. The
    // two differ by exactly the slack that buys the reuse: capacity is kept at
    // the high watermark a series of shapes reached, so a later smaller shape
    // fits without asking for memory again.
    size_t getActivationPeak() const { return peak; }

    size_t getActivationCapacity() const {
        if (hasMemPool)
            return memPoolSize;
        return ptr ? ptr->getBytes() : 0;
    }

    size_t getActivationAllocations() const { return activationAllocations; }

    // void addCache(size_t batchsize, std::unordered_map<TensorObj *, size_t>);

    // std::unordered_map<TensorObj *, size_t> getCache(size_t batchsize);

    Blob getActivationBlob(size_t offset, size_t bytes);

    Blob getActivationBlob(const Blob &storage, size_t offset,
                           size_t bytes) const;

    void *getWeightPtr();

    Blob getWeightBlob(size_t offset, size_t bytes);

    void *getHeapPtr();

    Blob getHeapBlob(size_t offset, size_t bytes);

    void info();

  private:
    // function: memory alignment, rouned up
    // return: size of the aligned memory block
    size_t getAlignedSize(size_t size);
};

} // namespace infini
