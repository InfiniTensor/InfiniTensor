#pragma once
#include "code_gen/nnet/Pass/Pass.h"
#include "code_gen/nnet/ReplaceKit.h"

namespace nnet {

class Rule8GuidedDLT : public Pass {
  public:
    Rule8GuidedDLT(Derivator &derivator) : Pass(derivator, "Rule8GuidedDLT") {}
    VecExpr guidedDLT(Formula &origin, int depth, Expr &rCur,
                      bool debug = false);

  private:
    virtual void transform(Formula &origin, int dfsDepth, Expr &rCur) override;
    /**
     * @brief If only one row miss match (more iterators mismatch), directly do
     * data layout construction according to the IT.
     *
     * @return Expr Return nullptr if failed.
     */
    Expr guidedDLTMoreVar2(const RangeOp &cur, const Mismatch &mismatch,
                           const IteratorTable &exprIT, const Pattern &pattern);
    /**
     * @brief Check whether two iterators overlap each other. If overlapping, we
     * cannot simply reconstruct the tensor into a new one by seperate all
     * iterators into different dimensions.
     */
    bool checkElementsHaveOnlyOneAccessIteratorSet(const IteratorTable &exprIT,
                                                   int tensorID);
    /**
     * @brief Only product of two tensors can be guided DLTed.
     *
     * @param cur
     * @return true
     * @return false
     */
    bool statisfyGuidedDLT(RangeOp cur) const;
    /**
     * @brief Deal with output DLT mismatch only.
     */
    Expr guidedDLTDLMismatch(const RangeOp &cur, const Mismatch &mismatch,
                             const IteratorTable &exprIT,
                             const Pattern &pattern);
    Expr buildGuidedDLTSource(const Subscript &originalSub, Replace replace,
                              vector<Var> tensorDimAxes, vector<int> newShape);
};

} // namespace nnet