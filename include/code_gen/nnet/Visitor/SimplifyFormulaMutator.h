#pragma once
#include "code_gen/nnet/Visitor/StrideVisitor.h"
#include "code_gen/nnet/visitor.h"

namespace nnet {

// Simplify all indexes in subscripts in an expression tree
class SimplifyFormulaMutator : public Mutator {
    int nSubscripts = 0;

  public:
    SimplifyFormulaMutator() : Mutator(0) {}
    Expr visit_(const Subscript &c) override;
    // Expr visit_(const BinaryOp &c) override;
    Expr simplify(const Expr &expr);
};

} // namespace nnet