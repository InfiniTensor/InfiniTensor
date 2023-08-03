#include "core/blob.h"
#include "core/runtime.h"

namespace infini {

BlobObj::~BlobObj() {
    // Avoid cycled inclusion
    // destruction is performed in LazyAllocator
    // runtime->dealloc(ptr);
}

} // namespace infini