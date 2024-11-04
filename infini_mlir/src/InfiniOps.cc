#include "Dialect/InfiniOps.h"
#include "Dialect/InfiniOpsDialect.cpp.inc"
namespace infini {
namespace infinimlir {
void InfiniDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "Dialect/InfiniOps.cpp.inc"
        >();
}
} // namespace infinimlir
} // namespace infini
#define GET_OP_CLASSES
#include "Dialect/InfiniOps.cpp.inc"