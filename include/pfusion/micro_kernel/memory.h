#pragma once

#include "pfusion/micro_op.h"

namespace memb {
class MemoryOp : public MicroOp {
  private:
    OpType opType;
    std::shared_ptr<Pointer> src, dst;
    size_t num, width;

  public:
    MemoryOp(OpType _opType, std::shared_ptr<Pointer> _src,
             std::shared_ptr<Pointer> _dst, size_t _num, size_t _width)
        : opType(_opType), src(_src), dst(_dst), num(_num), width(_width) {}
    // bool checkValid() override;
    std::string generate() override;
    inline void print() override {
        if (opType == READ) {
            std::cout << id << " " << getName(opType) << " "
                      << getName(src->getType()) << std::endl;
        } else if (opType == WRITE) {
            std::cout << id << " " << getName(opType) << " "
                      << getName(dst->getType()) << std::endl;
        } else {
            IT_ASSERT(false);
        }
    }
};
} // namespace memb
