#include "core/lazy_allocator.h"

namespace infini {
    LazyAllocator::LazyAllocator(Runtime runtime) : runtime(runtime){}

    LazyAllocator::~LazyAllocator() {}

    void LazyAllocator::init(size_t size) {
        peak = size;
    }

    size_t LazyAllocator::alloc(size_t size, size_t alignment) {
        return 0;
    }

    void LazyAllocator::free(size_t addr) {

    }

    Blob LazyAllocator::ptr() {
        return runtime->allocBlob(peak);
    }

} // namespace infini