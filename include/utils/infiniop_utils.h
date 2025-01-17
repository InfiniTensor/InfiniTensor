#pragma once
#include "core/common.h"
#include "infini_operators.h"
#include <vector>

namespace infini {
#define CHECK_ERROR(call)                                                      \
    if (auto err = call; err != STATUS_SUCCESS)                                \
    throw ::infini::Exception(                                                 \
        std::string("[") + __FILE__ + ":" + std::to_string(__LINE__) +         \
        "] operators error (" + #call + "): " + std::to_string(err))

DataLayout toInfiniopDataLayout(int dataType);

std::vector<uint64_t> toInfiniopShape(const std::vector<int> &shape);
} // namespace infini
