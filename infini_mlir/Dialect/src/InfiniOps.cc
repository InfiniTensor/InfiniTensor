#include "InfiniOps.h"
#include "InfiniOpsDialect.cpp.inc"
#define GET_OP_CLASSES
#include "InfiniOps.cpp.inc"
namespace infini {
namespace infinimlir {
void InfiniDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "InfiniOps.cpp.inc"
        >();
}

int64_t ConstantOp::getBitWidth() {
    mlir::Type t = getResult().getType();
    if (auto intType = t.dyn_cast<mlir::IntegerType>()) {
        return intType.getWidth();
    } else if (auto floatType = t.dyn_cast<mlir::FloatType>()) {
        return floatType.getWidth();
    } else if (auto fp16Type = t.dyn_cast<mlir::Float16Type>()) {
        return 16;
    } else {
        return 8;
    }
}

} // namespace infinimlir
} // namespace infini
