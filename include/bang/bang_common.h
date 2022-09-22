#pragma once
#include "cnnl.h"
#include "cnrt.h"
#include "core/common.h"

#define checkBangError(call)                                                   \
    {                                                                          \
        auto err = call;                                                       \
        if (CNRT_RET_SUCCESS != err) {                                         \
            fprintf(stderr, "Bang error in %s:%i : %s.\n", __FILE__, __LINE__, \
                    cnrtGetErrorStr(err));                                     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

#define checkCnnlError(call)                                                   \
    {                                                                          \
        auto err = call;                                                       \
        if (CNNL_STATUS_SUCCESS != err) {                                      \
            fprintf(stderr, "cnnl error in %s:%i : %s.\n", __FILE__, __LINE__, \
                    cnnlGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

namespace infini {

using BangPtr = void *;

} // namespace infini
