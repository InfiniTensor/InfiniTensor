#pragma once
#include "code_gen/nnet/Pass/Pass.h"

namespace nnet {

class Rule91MergeStagesWithSum : public Pass {
  public:
    Rule91MergeStagesWithSum(Derivator &derivator)
        : Pass(derivator, "Rule91MergeStagesWithSum") {}

  private:
    virtual void transform(Formula &origin, int dfsDepth, Expr &rCur) override;
};

} // namespace nnet