#pragma once
#include "code_gen/nnet/visitor.h"

namespace nnet {

class MatmulTransposeMutator : public Mutator {
    Derivator &derivator;

  public:
    MatmulTransposeMutator(Derivator &derivator)
        : Mutator(1), derivator(derivator) {}
    VecExpr transpose(const Tensor &tensor);

  private:
    Tensor transposeInput(const Tensor &tensor);
};

} // namespace nnet