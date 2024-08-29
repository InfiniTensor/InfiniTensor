#pragma once
#include "code_gen/nnet/Pass/Pass.h"

namespace nnet {

class Rule3StageSplit : public Pass {
  private:
    map<int, vector<Var>> substituteRules;

  public:
    Rule3StageSplit(Derivator &derivator)
        : Pass(derivator, "Rule3StageSplit") {}

  private:
    virtual void transform(Formula &origin, int dfsDepth, Expr &rCur) override;
    vector<vector<Var>> getSplitSummationIters(RangeOp rangeOp);
};

} // namespace nnet