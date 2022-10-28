#pragma once

#include "pfusion/micro_op.h"

namespace memb {
class UnaryOp : public MicroOp {
  private:
    const int num, width;

  public:
    UnaryOp(OpType _opType, std::shared_ptr<Pointer> _src,
            std::shared_ptr<Pointer> _dst, int _num, int _width)
        : num(_num), width(_width) {
        opType = _opType;
        ptrs = {_src, _dst};
    }
    ~UnaryOp() {}

    std::shared_ptr<Pointer> getSrc() { return ptrs[0]; }
    std::shared_ptr<Pointer> getDst() { return ptrs[1]; }

    // bool checkValid() override;
    std::string generate() override;
    inline void print() override {
        std::cout << id << " " << getName(opType) << std::endl;
    }
};

} // namespace memb