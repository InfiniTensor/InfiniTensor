#pragma once
#include "nnet/visitor.h"

namespace nnet {

class MatmulTransposeMutator : public Mutator {
    Derivator &derivator;

  public:
    MatmulTransposeMutator(Derivator &derivator)
        : Mutator(1), derivator(derivator) {}
    VecExpr transpose(const Tensor &tensor);

  private:
    optional<Tensor> transposeInput(const Tensor &tensor);
};

} // namespace nnet