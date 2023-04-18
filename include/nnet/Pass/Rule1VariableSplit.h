#pragma once
#include "nnet/Pass/Pass.h"
#include "nnet/ReplaceKit.h"

namespace nnet {

class Rule1VariableSplit : public Pass {
  public:
    Rule1VariableSplit(Derivator &derivator)
        : Pass(derivator, "Rule1VariableSplit") {}

  private:
    virtual void transform(Formula &origin, int dfsDepth, Expr &rCur) override;
    vector<Replace> getSplitableVar(const RangeOp &rangeOp);
    Expr replaceIters(Expr cur, const Replace &replace);
};

} // namespace nnet
