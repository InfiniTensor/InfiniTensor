#include "core/blob.h"
#include "core/runtime.h"

namespace infini {

BlobObj::~BlobObj() {
    // Avoid cycled inclusion
    // 析构在 LazyAllocator 中进行
    // runtime->dealloc(ptr);
}

} // namespace infini