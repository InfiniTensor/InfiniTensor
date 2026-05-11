#pragma once
#include "acl/acl.h"
#include "acl/acl_op.h"
#include "core/common.h"

#define checkASCENDError(call)                                                 \
    {                                                                          \
        auto err = call;                                                       \
        if (ACL_SUCCESS != err) {                                              \
            fprintf(stderr, "ASCEND error in %s:%i : .\n", __FILE__,           \
                    __LINE__);                                                 \
            auto tmp_err_msg = aclGetRecentErrMsg();                           \
            if (tmp_err_msg != NULL) {                                         \
                printf(" ERROR Message : %s \n ", tmp_err_msg);                \
            }                                                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

#define checkHCCLError(call)                                                   \
    {                                                                          \
        auto err = call;                                                       \
        if (HCCL_SUCCESS != err) {                                             \
            fprintf(stderr, "HCCL error in %s:%i : .\n", __FILE__, __LINE__);  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

#define GetRecentErrMsg()                                                      \
    {                                                                          \
        auto tmp_err_msg = aclGetRecentErrMsg();                               \
        if (tmp_err_msg != NULL) {                                             \
            printf(" ERROR Message : %s \n ", tmp_err_msg);                    \
        }                                                                      \
    }

namespace infini {

using ASCENDPtr = void *;

} // namespace infini
