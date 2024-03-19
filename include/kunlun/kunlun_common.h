#pragma once
#include "core/common.h"
#include "xpu/runtime_ex.h"
#include "xpu/xdnn.h"

namespace xdnn = baidu::xpu::api;

#define checkKUNLUNError(call)                                                 \
    {                                                                          \
        auto err = call;                                                       \
        if (XPU_SUCCESS != err) {                                              \
            fprintf(stderr, "KUNLUN error in %s:%i : %s.\n", __FILE__,         \
                    __LINE__, xpu_strerror(err));                              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

namespace infini {

using KUNLUNPtr = void *;

} // namespace infini
