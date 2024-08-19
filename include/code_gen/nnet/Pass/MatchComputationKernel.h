#pragma once
#include "code_gen/nnet/Pass/Pass.h"

namespace nnet {

class MatchComputationKernel : public Pass {
  public:
    MatchComputationKernel(Derivator &derivator)
        : Pass(derivator, "MatchComputationKernel") {}

  private:
    virtual void transform(Formula &origin, int dfsDepth, Expr &rCur) override;
};

} // namespace nnet