#pragma once
#include "core/common.h"
#include <musa.h>
#include <musa_runtime_api.h>

#define checkMusaError(call)                                                   \
    if (auto err = call; err != musaSuccess)                                   \
    throw ::infini::Exception(std::string("[") + __FILE__ + ":" +              \
                              std::to_string(__LINE__) + "] MUSA error (" +    \
                              #call + "): " + musaGetErrorString(err))

namespace infini {

using MusaPtr = void *;

class MUSAStream {
  public:
    MUSAStream(const MUSAStream &) = delete;
    MUSAStream(MUSAStream &&) = delete;
    void operator=(const MUSAStream &) = delete;
    void operator=(MUSAStream &&) = delete;
    static musaStream_t getCurrentStream() { return _stream; }
    static void Init() { MUSAStream::_stream = 0; }
    static void createStream() { checkMusaError(musaStreamCreate(&_stream)); }
    static void destroyStream() { checkMusaError(musaStreamDestroy(_stream)); }

  private:
    MUSAStream() {}
    static musaStream_t _stream;
};

} // namespace infini
