#pragma once
#include "code_gen/nnet/visitor.h"

namespace nnet {

class RangeMagnifyVisitor : public Mutator {
    vector<VarRangePair> newSumVarRanges;
    RangeOp newRangeOp;

  public:
    RangeMagnifyVisitor() : Mutator(0) {}
    Expr visit_(const RangeOp &c) override;
    Expr visit_(const Subscript &c) override;
    /**
     * @brief
     *
     * @param root
     * @param _newSumVarRanges
     * @return RangeOp nullptr if failed to magnify
     */
    RangeOp magnify(const RangeOp &root,
                    const vector<VarRangePair> &_newSumVarRanges);
};

} // namespace nnet