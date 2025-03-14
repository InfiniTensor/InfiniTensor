#include "core/dummy_mutator.h"
#include "operators/concat.h"
#include "operators/conv.h"
#include "operators/matmul.h"
#include "operators/split.h"
#include "operators/unary.h"

namespace infini {

vector<Graph> DummyMutator::run(const Graph &inGraph) {
    if (inGraph->getOperators().size() > 1)
        return {inGraph};
    // Conv -> Conv + Relu
    auto op0 = as<ConvObj>(inGraph->getOperators()[0]);
    auto g = make_ref<GraphObj>(runtime);
    auto a0 = g->cloneTensor(op0->getInputs()[0]),
         w0 = g->cloneTensor(op0->getInputs()[1]),
         o0 = g->cloneTensor(op0->getOutput());
    auto [ph, pw, sh, sw, dh, dw] = op0->getPadStrideDilation();
    auto t = g->addOp<ConvObj>(a0, w0, nullptr, ph, pw, nullptr, sh, sw, dh, dw)
                 ->getOutput();
    g->addOpWithOutputs<ReluObj>(t, o0);
    return {inGraph, g};
}

vector<Graph> DummyMutator::mergeMultiBranch(const Graph &inGraph) {
    // Two Mamtul of the same shapes -> One Batched Matmul
    if (!isMultiBranchMergable(inGraph))
        return {};
    auto op0 = as<MatmulObj>(inGraph->getOperators()[0]);
    auto op1 = as<MatmulObj>(inGraph->getOperators()[1]);
    auto [b, m, n, k, transA, transB] = op0->getBMNKTransAB();
    auto g = make_ref<GraphObj>(runtime);
    auto a0 = g->cloneTensor(op0->getInputs()[0]),
         w0 = g->cloneTensor(op0->getInputs()[1]),
         o0 = g->cloneTensor(op0->getOutput());
    auto a1 = g->cloneTensor(op1->getInputs()[0]),
         w1 = g->cloneTensor(op1->getInputs()[1]),
         o1 = g->cloneTensor(op1->getOutput());
    auto a = g->addOp<ConcatObj>(TensorVec{a0, a1}, nullptr, 0)->getOutput();
    auto w = g->addOp<ConcatObj>(TensorVec{w0, w1}, nullptr, 0)->getOutput();
    auto t = g->addOp<MatmulObj>(a, w, nullptr, transA, transB);
    g->addOpWithOutputs<SplitObj>(t->getOutput(), TensorVec{o0, o1}, 0, 2);
    return {g};
}

bool DummyMutator::isMultiBranchMergable(const Graph &inGraph) {
    if (inGraph->getOperators().size() != 2)
        return false;
    for (auto op : inGraph->getOperators()) {
        if (op->getOpType() != OpType::MatMul)
            return false;
        if (op->getPredecessors().size() > 0)
            return false;
        if (op->getSuccessors().size() > 0)
            return false;
    }
    auto op0 = as<MatmulObj>(inGraph->getOperators()[0]);
    auto op1 = as<MatmulObj>(inGraph->getOperators()[1]);
    auto args0 = op0->getBMNKTransAB();
    auto args1 = op1->getBMNKTransAB();
    return args0 == args1;
}

} // namespace infini
