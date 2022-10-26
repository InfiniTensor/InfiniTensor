#pragma once

#include "pfusion/common.h"
#include "pfusion/meta_op.h"

namespace memb {
std::vector<std::shared_ptr<MetaOp>>
instantiateUnary(const OpType opType,
                 std::vector<std::shared_ptr<Pointer>> ptrs,
                 const std::vector<size_t> &shape);
std::vector<std::shared_ptr<MetaOp>>
instantiateBinary(const OpType opType,
                  std::vector<std::shared_ptr<Pointer>> ptrs,
                  const std::vector<size_t> &shape);
std::vector<std::shared_ptr<MetaOp>> instantiateTranspose(
    const OpType opType, std::vector<std::shared_ptr<Pointer>> ptrs,
    const std::vector<size_t> &shape, const std::vector<size_t> &perm);
} // namespace memb
