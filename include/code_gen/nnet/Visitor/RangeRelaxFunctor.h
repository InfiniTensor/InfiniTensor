#pragma once
#include "code_gen/nnet/visitor.h"

namespace nnet {

using RangeMap = PtrMap<Iterator, Range>;
class RangeRelaxFunctor : public Functor<RangeMap()> {
    RangeOp rangeOp;

  public:
    RangeRelaxFunctor(RangeOp _rangeOp) : Functor(false), rangeOp(_rangeOp) {}
    RangeMap visit_(const BinaryOp &c) override;
    RangeMap visit_(const RangeOp &c) override;
    RangeMap visit_(const Subscript &c) override;
    RangeMap intersectRangeMaps(const RangeMap &a, const RangeMap &b);
};

} // namespace nnet