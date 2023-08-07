#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/unary.h"
#include "cuda/cuda_runtime.h"

#include "test.h"

namespace infini {

void testLazyAllocator(const Shape &shape, Runtime runtime) {
    Tensor a = make_ref<TensorObj>(shape, DataType::Float32, runtime);
    Tensor b = make_ref<TensorObj>(shape, DataType::Float32, runtime);
    Tensor c = make_ref<TensorObj>(shape, DataType::Float32, runtime);
    Tensor d = make_ref<TensorObj>(shape, DataType::Float32, runtime);
    LazyAllocator allocator = LazyAllocator(runtime);
    // allocator a->b->c->d
    allocator.alloc(a->getBytes());
    size_t offsetB = allocator.alloc(b->getBytes());
    size_t offsetC = allocator.alloc(c->getBytes());
    allocator.alloc(d->getBytes());
    // free b and c
    allocator.free(offsetB, b->getBytes());
    allocator.free(offsetC, c->getBytes());
    // expected to be a->mergedFreeBlock->d, where mergedFreeBlock is the result of merging the memory blocks corresponding to the already freed b and c
    EXPECT_EQ(allocator.freeBlocks.size(), 1);
    EXPECT_EQ(allocator.freeBlocks.begin()->addr, offsetB);
    EXPECT_EQ(allocator.freeBlocks.begin()->blockSize, allocator.getAlignedSize(b->getBytes()) + allocator.getAlignedSize(c->getBytes()));
}

TEST(Lazy_Allocator, runOnCpu) {
    testLazyAllocator(Shape{1, 2, 2, 3}, NativeCpuRuntimeObj::getInstance());
}

TEST(Lazy_Allocator, runOnGpu) {
    testLazyAllocator(Shape{1, 2, 2, 3}, make_ref<CudaRuntimeObj>());
}

} // namespace infini