#pragma once
#include "code_gen/nnet/Pass/Pass.h"
#include "code_gen/nnet/ReplaceKit.h"

namespace nnet {

class Rule2VariableMerging : public Pass {
  private:
    map<int, vector<Var>> substituteRules;

  public:
    Rule2VariableMerging(Derivator &derivator)
        : Pass(derivator, "Rule2VariableMerging") {}

  private:
    virtual void transform(Formula &origin, int dfsDepth, Expr &rCur) override;

    vector<Replace> getMergableReplaces(RangeOp rangeOp, int depth);
    optional<Replace> getReplaceMergingTwoLoopIters(const RangeOp &rangeOp,
                                                    pair<Iterator, int> pairA,
                                                    pair<Iterator, int> pairB,
                                                    const IteratorTable &exprIT,
                                                    int tensorID);
    optional<Replace> getReplaceMappingTwoLoopIters(const RangeOp &rangeOp,
                                                    pair<Iterator, int> pa,
                                                    pair<Iterator, int> pb);
};

} // namespace nnet