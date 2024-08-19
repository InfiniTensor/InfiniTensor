#include "code_gen/nnet/Pass/MatchMemBoundKernel.h"
#include "code_gen/nnet/Visitor/InputVisitor.h"

namespace nnet {

void MatchMemBoundKernel::transform(Formula &origin, int depth, Expr &rCur) {
    // FIXME: Whether the Formula is a Membound OP should be checked.
    nnet_assert(derivator.getSearchState() == 3, __LINE__);
    nnet_assert(origin.root.get() == rCur.get(),
                "Only match the entire formula as a Membound Op");
    auto rangeOp = as<RangeOpNode>(origin.root);
    const auto &inputs = InputVisitor().getInputs(rangeOp);
    auto source =
        make_ref<ElementWiseNode>(rangeOp, inputs, rangeOp->getOutputShape());
    auto tensor =
        makeTensor(newTensorName(), rangeOp->getOutputShape(), {}, source);
    // The original code directly appends candidate. But it seems should be done
    // by the search.
    // appendCanddiate(as<TensorNode>(tensor), depth);
    nextStep(origin, depth, rCur, tensor);
}

} // namespace nnet