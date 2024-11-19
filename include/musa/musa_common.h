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

} // namespace infini
