#include "core/blob.h"
#include "core/runtime.h"

namespace infini {

BlobObj::~BlobObj() {
    // Avoid cycled inclusion
    runtime->dealloc(ptr);
}

} // namespace infini