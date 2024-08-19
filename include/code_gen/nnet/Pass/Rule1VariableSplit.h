#pragma once
#include "code_gen/nnet/Pass/Pass.h"
#include "code_gen/nnet/ReplaceKit.h"

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