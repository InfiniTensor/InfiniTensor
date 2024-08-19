#pragma once
#include "code_gen/nnet/visitor.h"

namespace nnet {

// Clone ExprNodes in a stage except Tensor, Var, and Constant nodes.
class CloneMutator : public Mutator {
  public:
    CloneMutator() : Mutator(false) {}
    Expr visit_(const Constant &c) override;
    Expr visit_(const Var &c) override;
    Expr visit_(const Tensor &c) override;
    Expr clone(const Expr &c) { return dispatch(c); }
};

} // namespace nnet