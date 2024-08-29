#include "code_gen/nnet/Visitor/MatmulTransposeMutator.h"

namespace nnet {

VecExpr MatmulTransposeMutator::transpose(const Tensor &tensor) {
    assert(tensor->getDims() == 2);
    const auto matmul = as<MatmulNode>(tensor->getSource());
    VecExpr ret;
    for (int i = 1; i < 8; ++i) {
        // Whether really transpose/swap AB. transa/b are the arguments for gemm
        const int Atrans = (i & 1) > 0;
        const int Btrans = (i & 2) > 0;
        const int ABswap = (i & 4) > 0;

        auto newShape = tensor->getShape();
        auto newPaddings = tensor->getPaddings();
        auto [b, m, n, k, transa, transb] = matmul->getArgs();
        auto inputs = matmul->getInputs();
        transa ^= Atrans;
        transb ^= Btrans;
        // build input transpose
        if (Atrans)
            inputs[0] = transposeInput(inputs[0]);
        if (Btrans)
            inputs[1] = transposeInput(inputs[1]);
        if (ABswap) {
            std::swap(inputs[0], inputs[1]);
            std::swap(m, n);
            std::swap(transa, transb);
            std::swap(newShape[0], newShape[1]);
            std::swap(newPaddings[0], newPaddings[1]);
            transa ^= 1;
            transb ^= 1;
        }
        dbg(inputs);
        // build new Gemm Routine and Tensor
        // HACK: trivially wrap the source to generate different hash
        auto _va = make_ref<VarNode>("transA");
        auto _vb = make_ref<VarNode>("transB");
        auto _vc = make_ref<VarNode>("swapAB");
        auto fakeSub = makeSubscript(matmul->getExpr(), {_va, _vb});
        auto fakeRangeWrapperForHackHash =
            makeRangeOperator({{_va, {0, Atrans + 100}},
                               {_vb, {0, Btrans + 100}},
                               {_vc, {0, ABswap + 100}}},
                              {}, fakeSub);
        Matmul newMatmul =
            make_ref<MatmulNode>(fakeRangeWrapperForHackHash, inputs[0],
                                 inputs[1], b, m, n, k, transa, transb);
        auto newTensor = makeTensor(derivator.newTensorName(), newShape,
                                    newPaddings, newMatmul);
        // build output transpose
        if (ABswap) {
            vector<Var> vars{derivator.getNewVar(), derivator.getNewVar()};
            auto sub = makeSubscript(newTensor, {vars[1], vars[0]});
            vector<VarRangePair> loopVRs;
            // Sicne inputs array may be swaped, use the orignal tensor shape
            for (int i = 0; i < 2; ++i) {
                loopVRs.emplace_back(vars[i], Range(0, tensor->getShape(i)));
            }
            auto rangeOp = makeRangeOperator(loopVRs, {}, sub);
            ret.emplace_back(rangeOp);
        } else
            ret.emplace_back(newTensor);
    }
    return ret;
}

Tensor MatmulTransposeMutator::transposeInput(const Tensor &tensor) {
    Tensor ret;
    if (auto ew = as<ElementWiseNode>(tensor->getSource())) {
        auto rangeOp = as<RangeOpNode>(tensor->getSource()->getExpr());
        assert(rangeOp);
        assert(rangeOp->getNumOutputDims() == 2);
        auto loopVRs = rangeOp->getLoopVarRanges();
        std::swap(loopVRs[0], loopVRs[1]);
        // If there are paddings, the inner stage paddings should be removed
        assert(!rangeOp->hasPaddings());
        // auto paddings = rangeOp->getPaddings();
        // std::swap(paddings[0], paddings[1]);
        auto sub = makeSubscript(rangeOp, {loopVRs[1].first, loopVRs[0].first});
        auto newRangeOp = makeRangeOperator(loopVRs, {}, sub);
        // ElementWise newElementWise = make_ref<ElementWiseNode>(*ew);
        auto outputShape = ew->getOutputShape();
        std::swap(outputShape[0], outputShape[1]);
        auto newElementWise =
            make_ref<ElementWiseNode>(newRangeOp, ew->getInputs(), outputShape);

        auto tensorShape = tensor->getShape();
        auto tensorPaddings = tensor->getPaddings();
        std::swap(tensorShape[0], tensorShape[1]);
        std::swap(tensorPaddings[0], tensorPaddings[1]);
        ret = makeTensor(derivator.newTensorName(), tensorShape, tensorPaddings,
                         newElementWise);
        // } else if (!tensor->getSource()) {
    } else
        nnet_unimplemented_halt();
    return ret;
}

} // namespace nnet