#pragma once

#include "pfusion/micro_op.h"

namespace memb {
class BinaryOp : public MicroOp {
  private:
    OpType opType;
    std::shared_ptr<Pointer> pSrc0, pSrc1, pDst;
    size_t num, width;

  public:
    BinaryOp(OpType _opType, std::shared_ptr<Pointer> _pSrc0,
             std::shared_ptr<Pointer> _pSrc1, std::shared_ptr<Pointer> _pDst,
             size_t _num, size_t _width)
        : opType(_opType), pSrc0(_pSrc0), pSrc1(_pSrc1), pDst(_pDst), num(_num),
          width(_width) {}
    ~BinaryOp() {}
    // bool checkValid() override;
    std::string generate() override;
    inline void print() override {
        std::cout << id << " " << getName(opType) << std::endl;
    }
};

} // namespace memb