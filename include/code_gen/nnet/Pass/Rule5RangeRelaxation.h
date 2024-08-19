#pragma once
#include "code_gen/nnet/Pass/Pass.h"

namespace nnet {

class Rule5RangeRelaxation : public Pass {
  public:
    Rule5RangeRelaxation(Derivator &derivator)
        : Pass(derivator, "Rule5RangeRelaxation") {}
    Expr rule5RangeRelaxation(Formula &origin, int depth, Expr &rCur);

  private:
    virtual void transform(Formula &origin, int depth, Expr &rCur) override;
};

} // namespace nnet