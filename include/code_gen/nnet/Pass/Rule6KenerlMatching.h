#pragma once
#include "code_gen/nnet/Pass/Pass.h"

namespace nnet {

class Rule6KenerlMatching : public Pass {
  public:
    Rule6KenerlMatching(Derivator &derivator)
        : Pass(derivator, "Rule6KenerlMatching") {}

  private:
    virtual void transform(Formula &origin, int dfsDepth, Expr &rCur) override;
    // RE: seperating this func is a choice.
    VecExpr matchElementWise(const RangeOp &rangeOp);
};

} // namespace nnet