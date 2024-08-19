#pragma once
#include "code_gen/nnet/Visitor/StrideVisitor.h"
#include "code_gen/nnet/visitor.h"

namespace nnet {

class MatchTableVisitor
    : public Functor<void(const Tensor &, int dim, optional<int> stride)> {
  private:
    // Var -> {(tensor, dim)}
    Appearance appearance;
    vector<Tensor> tensors;
    vector<Subscript> subscripts;
    StrideTable strideTable;
    PtrMap<Iterator, vector<vector<int>>>
        strideInDim; // [Iterator][tensorID][dim]=stride

    // Intermediate variable
    // product of a sub-exprtree: Stride has to be done in two DFS
    SubexprSride subexprStride;
    bool hasUnsupportedOp = false;

  public:
    MatchTableVisitor(int _verobse = 0) : Functor(_verobse) {}
    void visit_(const BinaryOp &c, const Tensor &tensor, int dim,
                optional<int> stride) override;
    void visit_(const Subscript &c, const Tensor &tensor, int dim,
                optional<int> stride) override;
    void visit_(const Var &c, const Tensor &tensor, int dim,
                optional<int> stride) override;
    void visit_(const Constant &c, const Tensor &tensor, int dim,
                optional<int> stride) override;
    // void visit_(const Tensor &c, const Tensor &tensor) override;

    [[nodiscard]] bool operator()(const RangeOp &e) {
        hasUnsupportedOp = false;
        // get the location and stride of each iterator
        auto mulOp = as<BinaryOpNode>(e->getSummand());
        // TODO [feature]: support complex index exprs
        if (!mulOp || mulOp->getOpType() != OpType::Mul) {
            nnet_unimplemented_continue();
            return false;
        }
        StrideVisitor strideVisitor(0);
        subexprStride = strideVisitor.getFormulaStride(e);
        dispatch(mulOp->getLhs(), nullptr, 0, 0);
        dispatch(mulOp->getRhs(), nullptr, 0, 0);
        subscripts.emplace_back(as<SubscriptNode>(mulOp->getLhs()));
        subscripts.emplace_back(as<SubscriptNode>(mulOp->getRhs()));
        assert(tensors.size() == subscripts.size());
        assert(tensors.size() < 5);
        return !hasUnsupportedOp;
    }

    auto getResult() const {
        return tuple(appearance, tensors, strideTable, subscripts);
    }
};

} // namespace nnet