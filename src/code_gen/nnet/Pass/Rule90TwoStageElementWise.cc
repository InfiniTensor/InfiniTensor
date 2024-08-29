#include "code_gen/nnet/Pass/Rule90TwoStageElementWise.h"
#include "code_gen/nnet/Visitor/InputVisitor.h"

namespace nnet {

void Rule90TwoStageElementWise::transform(Formula &origin, int depth,
                                          Expr &rCur) {
    auto cur = as<RangeOpNode>(rCur);
    { // Match element-wise OP
        auto replaces = matchTwoStageElementWise(cur);
        if (!replaces.empty())
            dbg(rCur);
        // dbg(replaces);
        for (auto newCur : replaces) {
            string msg = "====== END rule90TwoStageElementWise\n";
            dbg(msg);
            nextStep(origin, depth, rCur, newCur);
        }
    }
}

VecExpr
Rule90TwoStageElementWise::matchTwoStageElementWise(const RangeOp &rangeOp) {
    // If the stage is compute bound, then do not convert it.
    int64_t flops = rangeOp->getFlops(), outputSize = rangeOp->getOutputSize();
    int64_t inputSize = rangeOp->getInputSize(rangeOp);
    if (double(flops) / (inputSize + outputSize) > 3)
        return {};
    auto outerSub = as<SubscriptNode>(rangeOp->getSummand());
    if (!outerSub)
        return {};
    auto innerRangeOp = as<RangeOpNode>(outerSub->getObject());
    if (!innerRangeOp)
        return {};
    auto innerSub = as<SubscriptNode>(innerRangeOp->getSummand());
    if (!innerSub)
        return {};
    auto innerTensor = as<TensorNode>(innerSub->getObject());
    if (!innerTensor)
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