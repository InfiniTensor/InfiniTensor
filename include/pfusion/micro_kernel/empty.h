#pragma once

#include "pfusion/micro_op.h"

namespace memb {
class EmptyOp : public MicroOp {
  public:
    EmptyOp() { opType = EMPTY; }
    ~EmptyOp() {}
    // bool checkValid() override;
    std::string generate() override { return ""; };
    inline void print() override {
        std::cout << id << " " << getName(opType) << std::endl;
    }
};

} // namespace memb