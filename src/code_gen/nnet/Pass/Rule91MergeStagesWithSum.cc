#include "code_gen/nnet/Pass/Rule91MergeStagesWithSum.h"
#include "code_gen/nnet/Pass/Rule4StageMerging.h"

namespace nnet {

void Rule91MergeStagesWithSum::transform(Formula &origin, int depth,
                                         Expr &rCur) {
    Rule4StageMerging(derivator).rule4StageMerging(origin, depth, rCur, true);
}

} // namespace nnet