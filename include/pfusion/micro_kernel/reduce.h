#pragma once

#include "pfusion/micro_op.h"

namespace memb {
class ReduceOp : public MicroOp {
  private:
    size_t num, width;

  public:
    ReduceOp(OpType _opType, std::shared_ptr<Pointer> _pSrc,
             std::shared_ptr<Pointer> _pDst, std::shared_ptr<Pointer> _pBuf,
             size_t _num, size_t _width)
        : num(_num), width(_width) {
        opType = _opType;
        ptrs = {_pSrc, _pDst, _pBuf};
    }
    ~ReduceOp() {}
    std::shared_ptr<Pointer> getSrc() { return ptrs[0]; }
    std::shared_ptr<Pointer> getDst() { return ptrs[1]; }
    std::shared_ptr<Pointer> getBuf() { return ptrs[2]; }
    // bool checkValid() override;
    std::string generate() override;
    inline void print() override {
        std::cout << id << " " << getName(opType) << std::endl;
    }
};

} // namespace memb