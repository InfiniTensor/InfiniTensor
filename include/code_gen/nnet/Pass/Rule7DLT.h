#pragma once
#include "code_gen/nnet/Pass/Pass.h"

namespace nnet {

class Rule7DLT : public Pass {
  public:
    Rule7DLT(Derivator &derivator) : Pass(derivator, "Rule7DLT") {}

  private:
    virtual void transform(Formula &origin, int dfsDepth, Expr &rCur) override;
    Expr buildDLTSingleRangeOp(const RangeOp &original, const Expr &newSummand);
    vector<int> getFactors();
};

} // namespace nnet