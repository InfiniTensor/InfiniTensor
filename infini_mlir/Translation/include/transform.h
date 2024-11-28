#pragma once
#include "InfiniOps.h"
#include "core/graph.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"

namespace infini {

namespace infinimlir {

class Transformation {
  public:
    Transformation(mlir::MLIRContext &context);
    Graph transform(GraphObj *graph);

  private:
    mlir::MLIRContext &context;
    mlir::PassManager pm;
};
} // namespace infinimlir

} // namespace infini