#include "code_gen/nnet/dlt.h"
#include "code_gen/nnet/Visitor/SimplifyExprVisitor.h"
#include "code_gen/nnet/visitor.h"
#include <algorithm>

namespace nnet {

optional<Expr> DLT::apply(const RangeOp &rangeOp, const Subscript &subscript,
                          string newTensorName) {
    const auto &tensor = as<TensorNode>(subscript->getObject());
    assert(tensor != nullptr);
    vector<int> shape0(tensor->getShape()), shape1;
    const auto &paddings = tensor->getPaddings();
    VecExpr index0 = subscript->getIndex(), index1;
    // compute new shapes and index
    for (const auto &opPtr : ops) {
        if (auto op = as<DLTSplit>(opPtr)) {
            assert(op->dim < (int)shape0.size());
            for (int i = 0; i < (int)shape0.size(); ++i)
                if (i != op->dim) {
                    shape1.emplace_back(shape0[i]);
                    index1.emplace_back(index0[i]);
                } else {
                    assert(shape0[i] % op->factor == 0);
                    shape1.emplace_back(shape0[i] / op->factor);
                    shape1.emplace_back(op->factor);
                    if (const auto &opt =
                            splitIndex(index0[i], op->factor, rangeOp);
                        opt.has_value()) {
                        index1.emplace_back(opt->first);
                        index1.emplace_back(opt->second);
                    } else
                        return {};
                }
        } else if (auto op = as<DLTMerge>(opPtr)) {
            assert(op->dim0 < (int)shape0.size());
            assert(op->dim1 < (int)shape0.size());
            for (int i = 0; i < (int)shape0.size(); ++i)
                if (i == op->dim0) {
                    shape1.emplace_back(shape0[op->dim0] * shape0[op->dim1]);
                    index1.emplace_back(index0[op->dim0] * shape0[op->dim1] +
                                        index0[op->dim1]);
                } else if (i != op->dim1) {
                    shape1.emplace_back(shape0[i]);
                    index1.emplace_back(index0[i]);
                }
        } else if (auto op = as<DLTReorder>(opPtr)) {
            if (op->dims.size() != shape0.size()) {
                // TODO: input Reorder should have the same order with tensor
                nnet_unimplemented_continue();
                return {};
            }
            assert(op->dims.size() == shape0.size());
            for (size_t i = 0; i < shape0.size(); ++i) {
                shape1.emplace_back(shape0[op->dims[i]]);
                index1.emplace_back(index0[op->dims[i]]);
            }
        }
        for (const auto &index : index1) {
            // Maybe there are bugs...
            // assert(index != nullptr);
            if (index == nullptr) {
                std::cout << "Warning empty" << std::endl;
                return {};
            }
        }
        shape0.swap(shape1);
        shape1.clear();
        index0.swap(index1);
        index1.clear();
    }
    for (auto &index : index0) {
        // Maybe there are bugs...
        assert(index != nullptr);
        if (index == nullptr)
            return {};
        index = SimplifyExprVisitor().simplify(index);
    }
    // HACK DLT with paddings: transfer original paddings to the new one
    vector<int> dltedPaddings =
        (paddings.size() == shape0.size()) ? paddings : vector<int>{};
    // TODO [necessary] build DLT source expr. Is OP-based DLT is good too?
    // HACK [important] fix this fake tensor.
    auto elementRoutine = make_ref<ElementWiseNode>(
        // FIXME: implement transpose
        // makeTensor(newTensorName + "_DLT", {}), vector<Tensor>{tensor},
        // shape0);
        makeTensor("__DLT", {}), vector<Tensor>{tensor}, shape0);
    auto dltedTensor =
        makeTensor(newTensorName, shape0, dltedPaddings, elementRoutine);
    auto dltedSubscript = makeSubscript(dltedTensor, index0);
    return optional<Expr>(std::in_place, dltedSubscript);
}

optional<pair<Expr, Expr>> DLT::splitIndex(Expr expr, int factor,
                                           RangeOp rangeOp) {
    auto strides = SimplifyExprVisitor().getStrides(expr);
    Expr quotient, remainder;
    for (const auto &[iter, stride] : strides) {
        const auto &[var, range] = rangeOp->getVarRange(iter);
        // Add new expr, dealing with the initial empty expr
        auto addExpr = [](Expr &orig, const Expr &newExpr) {
            if (!orig)
                orig = newExpr;
            else
                orig = orig + newExpr;
        };
        if (abs(stride) >= factor) {
            if (stride % factor)
                return {};
            addExpr(quotient, (stride / factor) * var);
        } else {
            if (stride * (range.second - range.first) > factor)
                return {};
            addExpr(remainder, stride * var);
        }
    }
    return optional<pair<Expr, Expr>>(std::in_place, quotient, remainder);
}

void DLT::split(int dim, int factor) {
    ops.emplace_back(make_ref<DLTSplit>(dim, factor));
}
void DLT::merge(int dim0, int dim1) {
    ops.emplace_back(make_ref<DLTMerge>(dim0, dim1));
}
void DLT::reorder(vector<int> dims) {
    ops.emplace_back(make_ref<DLTReorder>(dims));
}

} // namespace nnet
