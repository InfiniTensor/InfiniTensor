#pragma once

#include "InfiniOps.h"
#include "core/graph.h"
#include "core/runtime.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "operators/element_wise.h"

namespace infini {

namespace infinimlir {
Graph convertMLIRToInfini(mlir::ModuleOp, Runtime runtime);
void handleAddOp(Graph &g, mlir::Operation *op,
                 llvm::DenseMap<mlir::Value, Tensor> &tensorMap);
void handleConstantOp(Graph &g, mlir::Operation *op,
                      llvm::DenseMap<mlir::Value, Tensor> &tensorMap);
} // namespace infinimlir

} // namespace infini
