#pragma once

#include "pfusion/common.h"
#include "pfusion/meta_op.h"

namespace memb {
std::vector<std::shared_ptr<MetaOp>>
instantiateUnary(const std::vector<int> &shape, const OpType opType);
std::vector<std::shared_ptr<MetaOp>>
instantiateBinary(const std::vector<int> &shape, const OpType opType);
std::vector<std::shared_ptr<MetaOp>>
instantiateTranspose(const std::vector<int> &_shape,
                     const std::vector<int> &_perm);
} // namespace memb
