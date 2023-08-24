#pragma once
#include "core/common.h"
#include "xpu/runtime_ex.h"
#include "xpu/xdnn.h"

#define checkXPUError(call)                                                    \
    {                                                                          \
        auto err = call;                                                       \
        if (XPU_SUCCESS != err) {                                              \
            fprintf(stderr, "XPU error in %s:%i : %s.\n", __FILE__, __LINE__,  \
                    xpu_strerror(err));                                        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

namespace infini {

using XPUPtr = void *;

} // namespace infini
