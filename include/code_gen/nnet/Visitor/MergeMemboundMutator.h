#pragma once
#include "code_gen/nnet/visitor.h"

namespace nnet {

class MergeMemboundMutator : public Mutator {
    VecExpr kernels;
    int curDepth; // from the last one to the first one
    Expr visit_(const Tensor &c) override;
    // FIXME: duplicate code
    Expr rule4StageMerging(Expr &rCur, bool mergeStageWithCalc);
    bool checkEmpty();

  public:
    MergeMemboundMutator(const VecExpr &kernels)
        : Mutator(), kernels(kernels), curDepth(kernels.size() - 1) {}
    Expr merge(bool allowEmptyMembound = false);
};

} // namespace nnet