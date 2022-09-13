#pragma once
#include "nnet/visitor.h"

namespace nnet {

class MergeMemboundMutator : public Mutator {
    VecExpr kernels;
    int curDepth; // from the last one to the first one
    Expr visit_(const Tensor &c) override;
    // FIXME: duplicate code
    Expr rule4StageMerging(Expr &rCur, bool mergeStageWithCalc);
    bool checkEmpty();

  public:
    /**
     * @brief Construct a new Merge Membound Mutator object
     *
     * @param kernels Exprs in kernels are lsitded from inner to outer. The last
     * expr is the most outer one after merge.
     */
    MergeMemboundMutator(const VecExpr &kernels)
        : Mutator(), kernels(kernels), curDepth(kernels.size() - 1) {}
    Expr merge(bool allowEmptyMembound = false);
};

} // namespace nnet