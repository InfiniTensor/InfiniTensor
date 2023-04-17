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

    /// @brief Merged multiple expressions into one with one or several stages.
    /// @param allowEmptyMembound
    /// @param allowFailure If true, return nullptr when merging fails. If
    /// false, assert will fail.
    /// @return
    Expr merge(bool allowEmptyMembound = false, bool allowFailure = false);
};

} // namespace nnet
