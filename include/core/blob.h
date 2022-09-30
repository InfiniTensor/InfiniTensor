#pragma once
#include "core/common.h"
#include "core/ref.h"

namespace infini {

class RuntimeObj;
using Runtime = Ref<RuntimeObj>;

class BlobObj {
    // Runtime might be replaced with a raw pointer for optimization
    Runtime runtime;
    void *ptr;

  public:
    BlobObj(Runtime runtime, void *ptr) : runtime(runtime), ptr(ptr) {}
    BlobObj(BlobObj &other) = delete;
    BlobObj &operator=(BlobObj const &) = delete;
    ~BlobObj();

    template <typename T> T getPtr() const { return reinterpret_cast<T>(ptr); }
};

} // namespace infini
