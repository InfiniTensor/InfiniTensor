#pragma once
#include "code_gen/nnet/Pass/Pass.h"

namespace nnet {

class Rule90TwoStageElementWise : public Pass {
  public:
    Rule90TwoStageElementWise(Derivator &derivator)
        : Pass(derivator, "Rule90TwoStageElementWise") {}

  private:
    virtual void transform(Formula &origin, int dfsDepth, Expr &rCur) override;
    VecExpr matchTwoStageElementWise(const RangeOp &rangeOp);
};

} // namespace nnet