#pragma once
#include "core/common.h"
#include "core/ref.h"
#include <cstdint>

namespace infini {

class RuntimeObj;
using Runtime = Ref<RuntimeObj>;

class BlobObj {
    // Runtime might be replaced with a raw pointer for optimization
    Runtime runtime;
    void *ptr;
    size_t bytes;
    Ref<BlobObj> owner;
    uint64_t storageId;
    size_t storageOffset;

  public:
    BlobObj(Runtime runtime, void *ptr, size_t bytes);
    BlobObj(Ref<BlobObj> owner, size_t offset, size_t bytes);
    BlobObj(BlobObj &other) = delete;
    BlobObj &operator=(BlobObj const &) = delete;
    ~BlobObj() noexcept;

    size_t getBytes() const { return bytes; }
    uint64_t getStorageId() const { return storageId; }
    size_t getStorageOffset() const { return storageOffset; }
    template <typename T> T getPtr() const {
        IT_ASSERT(ptr != nullptr, "Blob has no backing memory");
        return reinterpret_cast<T>(ptr);
    }
};

} // namespace infini
