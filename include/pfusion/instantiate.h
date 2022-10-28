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
std::vector<std::shared_ptr<MetaOp>>
instantiateGather(const OpType opType,
                  const std::vector<std::shared_ptr<Pointer>> &ptrs,
                  const std::vector<size_t> &inputShape,
                  const std::vector<size_t> &indexShape,
                  const std::vector<size_t> &outputShape, const size_t axis);
std::vector<std::shared_ptr<MetaOp>>
instantiateReduce(const OpType opType,
                  const std::vector<std::shared_ptr<Pointer>> &ptrs,
                  const std::vector<size_t> &inputShape, const size_t axis);
std::vector<std::shared_ptr<MetaOp>> instantiateBroadcast(
    const OpType opType, const std::vector<std::shared_ptr<Pointer>> &ptrs,
    const std::vector<size_t> &inputShape, const size_t axis, const size_t num);

} // namespace memb
