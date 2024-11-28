#include "InfiniOps.h"
#include "InfiniOpsDialect.cpp.inc"
namespace infini {
namespace infinimlir {
void InfiniDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "InfiniOps.cpp.inc"
        >();
}
} // namespace infinimlir
} // namespace infini
#define GET_OP_CLASSES
#include "InfiniOps.cpp.inc"