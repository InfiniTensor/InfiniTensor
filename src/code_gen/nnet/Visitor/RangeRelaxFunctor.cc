#include "code_gen/nnet/Visitor/RangeRelaxFunctor.h"
#include "code_gen/nnet/Visitor/SimplifyExprVisitor.h"

namespace nnet {

RangeMap RangeRelaxFunctor::visit_(const BinaryOp &c) {
    if (verbose)
        dbg(*c);
    if (c->getOpType() == OpType::Mul)
        return intersectRangeMaps(dispatch(c->getLhs()), dispatch(c->getRhs()));
    nnet_unimplemented_halt();
    return RangeMap();
}
RangeMap RangeRelaxFunctor::visit_(const RangeOp &c) {
    if (verbose)
        dbg(*c);
    return dispatch(c->getSummand());
}
RangeMap RangeRelaxFunctor::visit_(const Subscript &c) {
    if (verbose)
        dbg(*c);
    RangeMap ret;
    const auto &tensor = as<TensorNode>(c->getObject());
    for (size_t i = 0; i < c->getDims(); ++i) {
        const int padding = tensor->getPadding(i);
        const int length = tensor->getShape(i);
        if (!padding)
            continue;
        // TODO assert access within padding+length
        // If the index is a single Var
        if (auto var = as<VarNode>(c->getIndex(i))) {
            ret = intersectRangeMaps(ret, {{var, {0, length}}});
        } else { // If the index is linear expression
            const auto &[strides, offset] =
                SimplifyExprVisitor().getStridesConstant(c->getIndex(i));
            // // Calculate the sum of ranges of all iters in negative value
            // Range allRange(-offset, -offset);
            // for (const auto &[iter, stride] : strides) {
            //     auto iterRange = rangeOp->getVarRange(iter).second;
            //     if (stride > 0) {
            //         allRange.first -= stride * (iterRange.second - 1);
            //         allRange.second -= stride * iterRange.first;
            //     } else {
            //         allRange.first += stride * iterRange.first;
            //         allRange.second += stride * (iterRange.second - 1);
            //     }
            //     dbg(iter, stride, iterRange, allRange);
            // }
            // dbg(allRange);
            // // Calculate the meaningful ranges for each iter
            // for (const auto &[iter, stride] : strides) {
            //     auto iterRange = rangeOp->getVarRange(iter).second;
            //     auto rangeExceptThis{allRange};
            //     if (stride > 0) {
            //         rangeExceptThis.first += stride * (iterRange.second - 1);
            //         rangeExceptThis.second += stride * iterRange.first;
            //     } else {
            //         rangeExceptThis.first -= stride * iterRange.first;
            //         rangeExceptThis.second -= stride * (iterRange.second -
            //         1);
            //     }
            //     // Meaningful calculation range for current iter
            //     int l, r;
            //     if (stride > 0) {
            //         // l = (0 - rangeExceptThis.second + stride - 1) /
            //         stride;
            //         // r = (length - rangeExceptThis.first) / stride;
            //         l = (0 - rangeExceptThis.second + stride - 1) / stride;
            //         r = (length - 1 - rangeExceptThis.first) / stride + 1;
            //     } else {
            //         nnet_unimplemented_continue();
            //         continue;
            //     }
            //     dbg(iter, stride, iterRange, l, r);
            //     ret = intersectRangeMaps(ret, {{iter, {l, r}}});
            // }
            // Calculate the sum of ranges of all iters in negative value
            Range allRange(offset, offset);
            for (const auto &[iter, stride] : strides) {
                auto iterRange = rangeOp->getVarRange(iter).second;
                if (stride > 0) {
                    allRange.first += stride * iterRange.first;
                    allRange.second += stride * (iterRange.second - 1);
                } else {
                    allRange.first += stride * (iterRange.second - 1);
                    allRange.second += stride * iterRange.first;
                }
                // dbg(iter, stride, iterRange, allRange);
            }
            // Calculate the meaningful ranges for each iter
            for (const auto &[iter, stride] : strides) {
                auto iterRange = rangeOp->getVarRange(iter).second;
                auto rangeExceptThis{allRange};
                if (stride > 0) {
                    rangeExceptThis.first -= stride * iterRange.first;
                    rangeExceptThis.second -= stride * (iterRange.second - 1);
                } else {
                    rangeExceptThis.first -= stride * (iterRange.second - 1);
                    rangeExceptThis.second -= stride * iterRange.first;
                }
                // Meaningful calculation range for current iter
                int l, r;
                if (stride > 0) {
                    // l = (0 - rangeExceptThis.second + stride - 1) / stride;
                    // r = (length - rangeExceptThis.first) / stride;
                    l = (0 - rangeExceptThis.second + stride - 1) / stride;
                    r = (length - 1 - rangeExceptThis.first) / stride + 1;
                } else {
                    nnet_unimplemented_continue();
                    continue;
                }
                ret = intersectRangeMaps(ret, {{iter, {l, r}}});
            }
        }
    }
    return ret;
}

RangeMap RangeRelaxFunctor::intersectRangeMaps(const RangeMap &a,
                                               const RangeMap &b) {
    RangeMap ret(a);
    for (const auto &[k, v] : b) {
        if (!ret.count(k))
            ret[k] = v;
        else {
            auto const &u = ret[k];
            ret[k] = {max(u.first, v.first), min(u.second, v.second)};
        }
    }
    return ret;
}

} // namespace nnet