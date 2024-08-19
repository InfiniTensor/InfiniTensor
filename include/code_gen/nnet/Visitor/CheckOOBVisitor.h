#pragma once
#include "code_gen/nnet/visitor.h"

namespace nnet {

class CheckOOBVisitor : public ExprTreeVisitor {
    RangeOp rangeOp;
    bool detect = false;

  public:
    CheckOOBVisitor(int _verobse = 0) : ExprTreeVisitor(1, 1, 0, 0, _verobse) {}
    void visit_(const Subscript &c) override;

    /**
     * @brief
     * @return true If there is OOB
     * @return false If there is no OOB
     */
    bool checkRangeOp(const RangeOp &_rangeOp);
};

} // namespace nnet