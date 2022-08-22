#include "core/run_enigne.h"

namespace infini {

BlobObj::~BlobObj() {
    // Avoid cycled inclusion
    runtime->dealloc(ptr);
}

} // namespace infini