#include "pfusion/micro_op.h"
#include "pfusion/micro_kernel/empty.h"
#include "pfusion/micro_kernel/memory.h"

namespace memb {
std::shared_ptr<MicroOp> MicroOp::merge(std::shared_ptr<MicroOp> op0,
                                        std::shared_ptr<MicroOp> op1) {
    if (op0->getType() == WRITE && op1->getType() == READ) {
        auto memOp0 = std::dynamic_pointer_cast<MemoryOp>(op0);
        auto memOp1 = std::dynamic_pointer_cast<MemoryOp>(op1);
        if (memOp0->getDst()->getHash() == memOp1->getSrc()->getHash()) {
            if (memOp0->getSrc()->getHash() == memOp1->getDst()->getHash()) {
                return std::make_shared<EmptyOp>();
            } else {
                // TODO: gen reg to reg.
                // IT_ASSERT(false);
            }
        }
    }
    return nullptr;
}
} // namespace memb
