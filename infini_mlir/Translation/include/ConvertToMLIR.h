#pragma once

#include "InfiniOps.h"
#include "core/operator.h"

namespace infini {

namespace infinimlir {

class OperationConverter {
  public:
    virtual mlir::Operation *
    convertToMLIR(mlir::OpBuilder &builder, const Operator &op,
                  const std::vector<mlir::Value> &inputs) = 0;
};

class AddConverter : public OperationConverter {
  public:
    mlir::Operation *
    convertToMLIR(mlir::OpBuilder &builder, const Operator &op,
                  const std::vector<mlir::Value> &inputs) override;
};
std::unique_ptr<OperationConverter> createConverter(OpType type);
mlir::Operation *convertOpToMLIR(mlir::OpBuilder &builder, const Operator &op,
                                 const std::vector<mlir::Value> &inputs);
} // namespace infinimlir

} // namespace infini