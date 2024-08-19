#include "code_gen/nnet/Visitor/SimplifyExprVisitor.h"
#include "code_gen/nnet/Visitor/StrideVisitor.h"

namespace nnet {

pair<PtrMap<Iterator, int>, int>
SimplifyExprVisitor::getStridesConstant(const Expr &expr) {
    constant = 0;
    divStrides.clear();
    modStrides.clear();
    subexprStride = StrideVisitor().getExprStride(expr);
    strides.clear();
    dispatch(expr, 1);
    return {strides, constant};
}

optional<Range> SimplifyExprVisitor::getExprRange(const Expr &expr,
                                                  const RangeOp &rangeOp) {
    constant = 0;
    divStrides.clear();
    modStrides.clear();
    subexprStride = StrideVisitor().getExprStride(expr);
    strides.clear();
    dispatch(expr, 1);
    // Skip if there is divide and modulo
    if (!divStrides.empty() || !modStrides.empty() || !divExprStrides.empty() ||
        !modExprStrides.empty())
        return {};
    Range ret{constant, constant + 1};
    for (const auto &[iter, stride] : strides) {
        const auto &[l, r] = rangeOp->getRange(iter);
        if (stride > 0) {
            ret.first += l * stride;
            ret.second += (r - 1) * stride;
        } else {
            ret.first += (r - 1) * stride;
            ret.second += l * stride;
        }
    }
    return ret;
}

PtrMap<Iterator, int> SimplifyExprVisitor::getStrides(const Expr &expr) {
    return getStridesConstant(expr).first;
}

int SimplifyExprVisitor::getConstant(const Expr &expr) {
    return getStridesConstant(expr).second;
}

Expr SimplifyExprVisitor::simplify(const Expr &expr) {
    getStrides(expr);
    Expr ret = nullptr;
    // merge divide and modulo items
    for (const auto &[iterDividerPair, divStride] : divStrides) {
        const auto &[iter, mod] = iterDividerPair;
        // mod < 0 is a marker for merging vars with negtive strides. In math,
        // divider < 0 is not well-defined for mod, so it should be exist in our
        // epxrs and is only a temporary state which must be simpilified now.
        if (mod < 0) { // must perfectly merged.
            const auto &modStride = modStrides[iterDividerPair];
            assert(divStride / abs(mod) == modStride);
            assert(divStride > 0);
            strides.try_emplace(iterDividerPair.first, 0);
            strides[iterDividerPair.first] += abs(divStride / mod);
            modStrides.erase(iterDividerPair);
        } else if (divStride % mod == 0 && modStrides.count(iterDividerPair)) {
            const auto &modStride = modStrides[iterDividerPair];
            if (divStride / mod == modStride) {
                strides.try_emplace(iterDividerPair.first, 0);
                strides[iterDividerPair.first] += divStride / mod;
                modStrides.erase(iterDividerPair);
            } else
                ret = ret + divStride * (iterDividerPair.first /
                                         iterDividerPair.second);
        } else
            ret = ret +
                  divStride * (iterDividerPair.first / iterDividerPair.second);
    }
    // remaining modulo items
    for (const auto &[iterDividerPair, stride] : modStrides) {
        ret = ret + stride * (iterDividerPair.first % iterDividerPair.second);
    }
    // normal constant*variable items
    for (const auto &[iter, stride] : strides) {
        if (stride == 0)
            continue;
        Expr subexpr;
        if (stride == 1)
            subexpr = iter;
        else
            subexpr = stride * iter;
        ret = (ret) ? ret + subexpr : subexpr;
    }
    // not perfectly nested divide and modulo items
    for (const auto &[iterDividerPair, stride] : divExprStrides) {
        ret = ret + stride * (iterDividerPair.first / iterDividerPair.second);
    }
    for (const auto &[iterDividerPair, stride] : modExprStrides) {
        ret = ret + stride * (iterDividerPair.first % iterDividerPair.second);
    }
    ret = ret + constant;
    return ret ? ret : make_ref<ConstantNode>(0);
}

void SimplifyExprVisitor::visit_(const BinaryOp &c, optional<int> stride) {
    if (verbose)
        dbg(c);
    if (c->getOpType() == OpType::Add) {
        dispatch(c->getLhs(), stride);
        dispatch(c->getRhs(), stride);
    } else if (c->getOpType() == OpType::Sub) {
        dispatch(c->getLhs(), stride);
        if (stride)
            *stride = -*stride;
        dispatch(c->getRhs(), stride);
    } else if (c->getOpType() == OpType::Mul) {
        const optional<int> &lStride = subexprStride[c->getLhs().get()];
        const optional<int> &rStride = subexprStride[c->getRhs().get()];
        optional<int> lCurStride =
            (stride && rStride) ? optional(*stride * *rStride) : nullopt;
        optional<int> rCurStride =
            (stride && lStride) ? optional(*stride * *lStride) : nullopt;
        dispatch(c->getLhs(), lCurStride);
        dispatch(c->getRhs(), rCurStride);
    } else if (c->getOpType() == OpType::Mod) {
        const auto &param = c->getModDivParameter();
        if (param.has_value()) {
            modStrides.try_emplace(*param, 0);
            modStrides[*param] += stride.value();
        } else {
            const auto &paramExpr = c->getModDivExpr();
            modExprStrides.try_emplace(paramExpr, 0);
            modExprStrides[paramExpr] += stride.value();
        }
    } else if (c->getOpType() == OpType::Div) {
        const auto &param = c->getModDivParameter();
        if (param.has_value()) {
            divStrides.try_emplace(*param, 0);
            divStrides[*param] += stride.value();
        } else {
            const auto &paramExpr = c->getModDivExpr();
            divExprStrides.try_emplace(paramExpr, 0);
            divExprStrides[paramExpr] += stride.value();
        }
    } else
        nnet_unimplemented_halt();
}
void SimplifyExprVisitor::visit_(const Var &c, optional<int> stride) {
    if (verbose)
        dbg(c);
    strides.try_emplace(c);
    strides[c] += stride.value();
}
void SimplifyExprVisitor::visit_(const Constant &c, optional<int> stride) {
    if (stride.has_value())
        constant += stride.value() * c->getValue();
}

} // namespace nnet