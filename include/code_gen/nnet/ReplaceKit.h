#pragma once
#include "code_gen/nnet/expr.h"

namespace nnet {

struct Replace {
    int iteratorType;
    vector<Var> oldIters; // i_1, ...
    vector<Var> newIters; // j_1, ...
    VecExpr phis;         // j_1=\phi_1(i_1, ...), not necessary for Sum iter
    VecExpr psis;         // i_1=\psi_1(j_1, ...)
    vector<VarRangePair> newVarRanges;

    bool isReplaced(Var var) const {
        for (const auto &iter : oldIters)
            if (iter->equal(var))
                return true;
        return false;
    }

    string toReadable() const {
        string ret = "Old iters: " + serializeVec(oldIters) +
                     ", new iters: " + serializeVec(newIters);
        ret += " phis: " + serializeVec(phis) + " psis: " + serializeVec(psis);
        return ret;
    }
};

class ReplaceKit {
  public:
    static RangeOp replaceRangeOpIterator(const RangeOp &rangeOp,
                                          const Replace &replace,
                                          const Expr &replacedSummand);
    static Subscript buildSubscirptForLoopVarReplace(const RangeOp &inner,
                                                     const Replace &replace);
    static RangeOp buildDLTOuterRangeOp(const RangeOp &original,
                                        const Subscript &subscriptedNewRangeOp);
    static Expr replaceMultipleExprs(const Expr &cur,
                                     const vector<Var> &patterns,
                                     const VecExpr &replacements,
                                     bool simplify = true);
    static Expr replaceExpr(const Expr &cur, const Expr &pattern,
                            const Expr &replacement);
};

} // namespace nnet