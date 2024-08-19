#include "code_gen/nnet/Pass/MatchComputationKernel.h"
#include "code_gen/nnet/Visitor/PatternMatcher.h"

namespace nnet {

// RE: is this duplicate with Rule6KenerlMatching?
void MatchComputationKernel::transform(Formula &origin, int depth, Expr &rCur) {
    nnet_assert(derivator.getSearchState() == 2, __LINE__);
    dbg(depth, rCur);
    auto cur = as<RangeOpNode>(rCur);
    // Build wrapper stages for enforce axis starts from 0
    PatternMatcher patternMatcher(derivator, cur);
    cur = patternMatcher.getOffsetCur();

    auto matches = patternMatcher.matchWithPattern(
        cur, getPattern(derivator.getTargetOp()));
    matches = patternMatcher.applyWrapper(matches);

    for (auto newCur : matches) {
        string msg = "====== END rule MatchComputationKernel\n";
        dbg(msg);
        derivator.setSearchState(3);
        nextStep(origin, depth, rCur, newCur);
        derivator.setSearchState(2);
    }
}

} // namespace nnet