#pragma once

#include "pfusion/micro_op.h"
#include <string>

namespace memb {
class UnaryOp : public MicroOp {
  private:
    const OpType opType;
    const std::shared_ptr<Pointer> src, dst;
    const int num, width;

  public:
    UnaryOp(OpType _opType, std::shared_ptr<Pointer> _src,
            std::shared_ptr<Pointer> _dst, int _num, int _width)
        : opType(_opType), src(_src), dst(_dst), num(_num), width(_width) {}
    ~UnaryOp() {}
    // bool checkValid() override;
    std::string generate() override;
    inline void print() override {
        std::cout << id << " " << getName(opType) << std::endl;
    }
};

} // namespace memb