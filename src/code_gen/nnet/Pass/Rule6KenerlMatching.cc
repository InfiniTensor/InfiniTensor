#include "code_gen/nnet/Pass/Rule6KenerlMatching.h"
#include "code_gen/nnet/Visitor/InputVisitor.h"
#include "code_gen/nnet/Visitor/PatternMatcher.h"

namespace nnet {

void Rule6KenerlMatching::transform(Formula &origin, int depth, Expr &rCur) {
    dbg(depth, rCur);
    auto cur = as<RangeOpNode>(rCur);
    // Build wrapper stages for enforce axis starts from 0
    PatternMatcher patternMatcher(derivator, cur);
    cur = patternMatcher.getOffsetCur();

    // Match matchable routines
    for (int i = 0; i < MatchableRoutineTypeCnt; ++i) {
        auto targetOp = idToRoutineType(i);
        // During guided search, only check the target OP
        if (derivator.getTargetOp() != RoutineType::NoneType &&
            derivator.getTargetOp() != targetOp)
            continue;
        auto replaces =
            patternMatcher.matchWithPattern(cur, getPattern(targetOp));
        replaces = patternMatcher.applyWrapper(replaces);
        for (auto newCur : replaces) {
            string msg = "====== END rule6KenerlMatching\n";
            dbg(msg);
            nextStep(origin, depth, rCur, newCur);
        }
    }
    { // Match element-wise OP
        auto replaces = matchElementWise(cur);
        if (!replaces.empty())
            dbg(rCur);
        // dbg(replaces);
        for (auto newCur : replaces) {
            string msg = "====== END rule6KenerlMatching\n";
            dbg(msg);
            nextStep(origin, depth, rCur, newCur);
        }
    }
}

VecExpr Rule6KenerlMatching::matchElementWise(const RangeOp &rangeOp) {
    // If the stage is compute bound, then do not convert it.
    int64_t flops = rangeOp->getFlops(), outputSize = rangeOp->getOutputSize();
    int64_t inputSize = rangeOp->getInputSize(rangeOp);
    if (double(flops) / (inputSize + outputSize) > 3)
        return {};

    vector<int> newShape;
    for (const auto &[var, range] : rangeOp->getLoopVarRanges()) {
        if (range.first != 0) {
            nnet_unimplemented_continue();
            return {};
        }
        newShape.emplace_back(range.second - range.first);
    }
    const auto &inputs = InputVisitor().getInputs(rangeOp);
    auto source =
        make_ref<ElementWiseNode>(rangeOp, inputs, rangeOp->getOutputShape());
    auto newTensor = makeTensor(newTensorName(), newShape, {}, source);
    return {newTensor};
}

} // namespace nnet