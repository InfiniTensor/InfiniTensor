#pragma once
#include "code_gen/nnet/Pass/Pass.h"

namespace nnet {

class Rule9RangeMagnify : public Pass {
  public:
    Rule9RangeMagnify(Derivator &derivator)
        : Pass(derivator, "Rule9RangeMagnify") {}

  private:
    virtual void transform(Formula &origin, int dfsDepth, Expr &rCur) override;
};

} // namespace nnet