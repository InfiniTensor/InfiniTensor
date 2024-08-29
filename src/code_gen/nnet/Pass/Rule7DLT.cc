#include "code_gen/nnet/Pass/Rule7DLT.h"
#include "code_gen/nnet/Visitor/ReplaceNodeMutator.h"
#include "code_gen/nnet/dlt.h"

namespace nnet {

void Rule7DLT::transform(Formula &origin, int depth, Expr &rCur) {
    auto cur = as<RangeOpNode>(rCur);
    if (!cur)
        return;
    auto op = as<BinaryOpNode>(cur->getSummand());
    if (!op)
        return;
    auto subs = {op->getLhs(), op->getRhs()};
    for (auto subExpr : subs) {
        auto sub = as<SubscriptNode>(subExpr);
        if (!sub)
            continue;
        auto tensor = as<TensorNode>(sub->getObject());
        if (!tensor)
            continue;
        // // HACK for G2BMM
        // if (tensor->getDims() != 3)
        //     continue;
        for (const auto factor : getFactors()) {
            for (int targetDim = 0; targetDim < tensor->getDims();
                 ++targetDim) {
                if (tensor->getShape(targetDim) % factor)
                    continue;
                // Debug hint for G2BMM
                if (derivator.getPassMode() == Derivator::PassMode::Debug) {
                    if (tensor->getShape(targetDim) != 10000)
                        continue;
                    assert(targetDim == 1);
                }
                DLT dlt;
                dlt.split(targetDim, factor);
                vector<int> newOrder(tensor->getDims() + 1);
                for (int i = 0; i < tensor->getDims() + 1; ++i)
                    newOrder[i] = i;
                newOrder[targetDim]++;
                newOrder[targetDim + 1]--;
                dlt.reorder(newOrder);
                dlt.merge(targetDim, targetDim + 1);
                if (auto opt = dlt.apply(cur, sub, newTensorName())) {
                    Expr newSummand = ReplaceNodeMutator().replace(
                        cur->getSummand(), sub.get(), *opt);
                    auto newCur = buildDLTSingleRangeOp(cur, newSummand);

                    // next searching step
                    string msg = "====== END rule7DLT\n";
                    dbg(msg);
                    nextStep(origin, depth, rCur, newCur);
                }
            }
        }
    }
}

Expr Rule7DLT::buildDLTSingleRangeOp(const RangeOp &original,
                                     const Expr &newSummand) {
    auto rangeOp = make_ref<RangeOpNode>(*original);
    rangeOp->setSummand(newSummand);
    return rangeOp;
}

vector<int> Rule7DLT::getFactors() {
    if (derivator.getPassMode() == Derivator::PassMode::Debug) {
        return {4};
    } else if (derivator.getPassMode() == Derivator::PassMode::Full) {
        return {3, 4};
    } else {
        nnet_unimplemented_halt();
        return {};
    }
}

} // namespace nnet