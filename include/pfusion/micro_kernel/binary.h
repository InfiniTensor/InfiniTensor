#pragma once

#include "pfusion/micro_op.h"

namespace memb {
class BinaryOp : public MicroOp {
  private:
    size_t num, width;

  public:
    BinaryOp(OpType _opType, std::shared_ptr<Pointer> _pSrc0,
             std::shared_ptr<Pointer> _pSrc1, std::shared_ptr<Pointer> _pDst,
             size_t _num, size_t _width)
        : num(_num), width(_width) {
        opType = _opType;
        ptrs = {_pSrc0, _pSrc1, _pDst};
    }
    ~BinaryOp() {}
    std::shared_ptr<Pointer> getSrc0() { return ptrs[0]; }
    std::shared_ptr<Pointer> getSrc1() { return ptrs[1]; }
    std::shared_ptr<Pointer> getDst() { return ptrs[2]; }
    // bool checkValid() override;
    std::string generate() override;
    inline void print() override {
        std::cout << id << " " << getName(opType) << std::endl;
    }
};

} // namespace memb