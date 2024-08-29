#pragma once
#include "code_gen/nnet/visitor.h"

namespace nnet {

/**
 * @brief Since the output positions of operators always start from 0, we have
 * to offset them if the the boundary expression of is not 0.
 */
class PatternMatcher : public Functor<void(void)> {
  private:
    Derivator &derivator;
    bool hasNonZeroRange;
    const RangeOp originalCur;

  public:
    PatternMatcher(Derivator &derivator, const RangeOp &cur);
    /**
     * @brief Get the Cur whose loop vars are all offset to [0, x). Since
     * operator outputs start from 0, RangeOp has to be aligned.
     */
    RangeOp getOffsetCur();
    /**
     * @brief Add outer RangeOp to map the original positions to the new
     * positions staring from 0.
     *
     * @param exprs Tensors from matched exprs
     */
    VecExpr applyWrapper(const VecExpr &exprs);

    VecExpr matchWithPattern(const RangeOp &rangeOp, const Pattern &pattern);

  private:
    VecExpr matchKernel(const Pattern &pattern, const RangeOp &rangeOp,
                        IteratorTable &exprIT);
    // get reverse tensor and iterator map ([pattern tensor/iter ID] ->
    // real)
    Expr matchKernelWithTensorMap(const Pattern &pattern,
                                  const RangeOp &rangeOp,
                                  IteratorTable &exprIT);
};

} // namespace nnet