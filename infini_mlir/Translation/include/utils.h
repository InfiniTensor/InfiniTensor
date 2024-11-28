#pragma once
#include "core/tensor.h"
#include "mlir/IR/BuiltinTypes.h"
namespace infini {
namespace infinimlir {
mlir::Type convertDataTypeToMlirType(mlir::MLIRContext *context,
                                     infini::DataType dtype);
DataType convertMlirTypeToDataType(mlir::Type type);
std::vector<int64_t> int_to_int64t(const std::vector<int> &shape);
std::vector<int> int64t_to_int(const std::vector<int64_t> &shape);

} // namespace infinimlir
} // namespace infini