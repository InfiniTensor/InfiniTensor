#pragma once
#include "code_gen/nnet/visitor.h"

namespace nnet {

// Replace node according to its address the summand
// Only subscript and tensor are supported now.
class ReplaceNodeMutator : public Mutator {
    int nSubscripts = 0;
    ExprNode *target;
    Expr replacement;

  public:
    ReplaceNodeMutator() : Mutator(0) {}
    Expr visit_(const Subscript &c) override;
    Expr visit_(const Tensor &c) override;
    Expr replace(const Expr &root, ExprNode *_target, const Expr &_replace);
};

} // namespace nnet