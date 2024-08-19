#pragma once
#include "code_gen/nnet/Visitor/StrideVisitor.h"
#include "code_gen/nnet/visitor.h"

namespace nnet {

// Simplify a index expression tree
class SimplifyExprVisitor : public Functor<void(optional<int> stride)> {
  private:
    SubexprSride subexprStride;
    int constant;
    PtrMap<Iterator, int> strides; // [var]=strides

    map<pair<Iterator, int>, int, RefValueLess<pair<Iterator, int>>> divStrides,
        modStrides; // 3*(i%8): [<i,8>]=3

    // For divde and modulo with expr as dividend: 3*((i+1)%8): [<i+1,8>]=3
    map<pair<Expr, int>, int, RefAddrLess<pair<Expr, int>>> divExprStrides,
        modExprStrides;

  public:
    SimplifyExprVisitor() : Functor(0) {}
    void visit_(const BinaryOp &c, optional<int> stride) override;
    void visit_(const Var &c, optional<int> stride) override;
    void visit_(const Constant &c, optional<int> stride) override;
    PtrMap<Iterator, int> getStrides(const Expr &expr);
    // TODO [refactor]: move this to SimplifyFormulaMutator as a member func
    // this class should be get coefficients in a expr
    Expr simplify(const Expr &expr);
    int getConstant(const Expr &expr);
    pair<PtrMap<Iterator, int>, int> getStridesConstant(const Expr &expr);
    optional<Range> getExprRange(const Expr &expr, const RangeOp &rangeOp);
    PtrMap<Iterator, int> getStrides() { return strides; }
    const auto &getDivStrides() { return divStrides; }
    const auto &getModStrides() { return modStrides; }
};

} // namespace nnet