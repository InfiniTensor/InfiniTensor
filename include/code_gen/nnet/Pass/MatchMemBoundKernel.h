#pragma once
#include "code_gen/nnet/Pass/Pass.h"

namespace nnet {

class MatchMemBoundKernel : public Pass {
  public:
    MatchMemBoundKernel(Derivator &derivator)
        : Pass(derivator, "MatchMemBoundKernel") {}

  private:
    virtual void transform(Formula &origin, int dfsDepth, Expr &rCur) override;
};

} // namespace nnet