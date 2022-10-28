#pragma once

#include "pfusion/micro_op.h"

namespace memb {
class MemoryOp : public MicroOp {
  private:
    size_t num, width;

  public:
    MemoryOp(OpType _opType, std::shared_ptr<Pointer> _src,
             std::shared_ptr<Pointer> _dst, size_t _num, size_t _width)
        : num(_num), width(_width) {
        opType = _opType;
        ptrs = {_src, _dst};
    }
    // bool checkValid() override;
    ~MemoryOp() {}
    std::shared_ptr<Pointer> getSrc() { return ptrs[0]; }
    std::shared_ptr<Pointer> getDst() { return ptrs[1]; }
    std::string generate() override;
    inline void print() override {
        if (opType == READ) {
            std::cout << id << " " << getName(opType) << " "
                      << getName(getSrc()->getType()) << " "
                      << getSrc()->getHash() << std::endl;
        } else if (opType == WRITE) {
            std::cout << id << " " << getName(opType) << " "
                      << getName(getDst()->getType()) << " "
                      << getDst()->getHash() << std::endl;
        } else {
            IT_ASSERT(false);
        }
    }
};
} // namespace memb
