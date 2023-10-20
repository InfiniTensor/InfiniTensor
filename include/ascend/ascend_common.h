#pragma once
#include "core/common.h"
#include "acl/acl.h"
#include "acl/acl_op.h"

#define checkASCENDError(call)                                                 \
    {                                                                          \
        auto err = call;                                                       \
        if (ACL_SUCCESS != err) {                                              \
            fprintf(stderr, "ASCEND error in %s:%i : .\n", __FILE__,           \
                    __LINE__);                                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

namespace infini {

using ASCENDPtr = void *;

} // namespace infini
