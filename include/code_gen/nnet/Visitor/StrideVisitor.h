#pragma once
#include "code_gen/nnet/visitor.h"

namespace nnet {

using SubexprSride = map<const ExprNode *, optional<int>>;
class StrideVisitor : public Functor<optional<int>(void)> {
  private:
    SubexprSride subexprStride;

  public:
    StrideVisitor(int _verobse = 0) : Functor(_verobse) {}
    optional<int> visit_(const BinaryOp &c) override;
    optional<int> visit_(const Subscript &c) override;
    optional<int> visit_(const Var &c) override;
    optional<int> visit_(const Constant &c) override;
    // void visit_(const Tensor &c, const Tensor &tensor) override;

    auto getFormulaStride(const RangeOp &e) {
        subexprStride.clear();
        // get the location and stride of each iterator
        auto mulOp = as<BinaryOpNode>(e->getSummand());
        // TODO [feature]: support complex index exprs
        if (!mulOp || mulOp->getOpType() != OpType::Mul)
            nnet_unimplemented_continue();
        dispatch(mulOp->getLhs());
        dispatch(mulOp->getRhs());
        return subexprStride;
    }

    [[nodiscard]] auto getExprStride(const Expr &e) {
        subexprStride.clear();
        dispatch(e);
        return subexprStride;
    }
};

} // namespace nnet