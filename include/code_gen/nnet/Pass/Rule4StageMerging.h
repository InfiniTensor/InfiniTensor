#pragma once
#include "code_gen/nnet/Pass/Pass.h"

namespace nnet {

class Rule4StageMerging : public Pass {
    bool success, mergeStageWithCalc;

  public:
    Rule4StageMerging(Derivator &derivator)
        : Pass(derivator, "Rule4StageMerging"), success(false),
          mergeStageWithCalc(false) {}
    bool rule4StageMerging(Formula &origin, int depth, Expr &rCur,
                           bool mergeStageWithCalc = false);
    bool isSuccessful();
    void setMergeStageWithCalc(bool value);

  private:
    virtual void transform(Formula &origin, int depth, Expr &rCur) override;
};

} // namespace nnet