#pragma once
#include "code_gen/nnet/visitor.h"

namespace nnet {

class CompareMultiFormulasVisitor : public ExprTreeVisitor {
    vector<VarRangePair> newSumVarRanges;
    RangeOp newRangeOp;

  public:
    CompareMultiFormulasVisitor() : ExprTreeVisitor() {}
    bool compare(const VecExpr &roots);
};

} // namespace nnet