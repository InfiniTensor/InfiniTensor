#include "core/blob.h"
#include "core/runtime.h"
#include <cstdio>

namespace infini {

BlobObj::BlobObj(Runtime runtime, void *ptr, size_t bytes)
    : runtime(std::move(runtime)), ptr(ptr), bytes(bytes) {
    IT_ASSERT(this->runtime != nullptr, "Blob requires a runtime");
    IT_ASSERT(ptr != nullptr, "Blob requires non-null backing memory");
}

BlobObj::BlobObj(Ref<BlobObj> owner, size_t offset, size_t bytes)
    : runtime(owner ? owner->runtime : nullptr), ptr(nullptr), bytes(bytes),
      owner(std::move(owner)) {
    IT_ASSERT(this->owner != nullptr, "Blob view requires an owner");
    IT_ASSERT(offset <= this->owner->bytes, "Blob view offset is out of range");
    IT_ASSERT(bytes <= this->owner->bytes - offset,
              "Blob view exceeds its owner");
    ptr = static_cast<uint8_t *>(this->owner->ptr) + offset;
}

BlobObj::~BlobObj() noexcept {
    if (!owner && ptr != nullptr) {
        try {
            runtime->dealloc(ptr);
        } catch (const std::exception &error) {
            std::fprintf(stderr, "Error in ~BlobObj: %s\n", error.what());
        } catch (...) {
            std::fputs("Unknown error in ~BlobObj\n", stderr);
        }
    }
}

} // namespace infini
