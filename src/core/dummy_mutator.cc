#include "core/dummy_mutator.h"
#include "operators/concat.h"
#include "operators/conv.h"
#include "operators/matmul.h"
#include "operators/split.h"
#include "operators/unary.h"

namespace infini {

vector<Graph> DummyMutator::run(const Graph &in_graph) {
    if (in_graph->getOperators().size() > 1)
        return {in_graph};
    // Conv -> Conv + Relu
    auto op0 = as<ConvObj>(in_graph->getOperators()[0]);
    auto g = make_ref<GraphObj>(runtime);
    auto a0 = g->cloneTensor(op0->getInputs()[0]),
         w0 = g->cloneTensor(op0->getInputs()[1]),
         o0 = g->cloneTensor(op0->getOutput());
    auto [ph, pw, sh, sw, dh, dw] = op0->getPadStrideDilation();
    auto t =
        g->addOp<ConvObj>(a0, w0, nullptr, ph, pw, sh, sw, dh, dw)->getOutput();
    g->addOpWithOutputs<ReluObj>(t, o0);
    return {in_graph, g};
}

vector<Graph> DummyMutator::fusion(const Graph &in_graph) {
    // Two Mamtul of the same shapes -> One Batched Matmul
    if (!isFusible(in_graph))
        return {};
    auto op0 = as<MatmulObj>(in_graph->getOperators()[0]);
    auto op1 = as<MatmulObj>(in_graph->getOperators()[1]);
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

bool DummyMutator::isFusible(const Graph &in_graph) {
    if (in_graph->getOperators().size() != 2)
        return false;
    for (auto op : in_graph->getOperators()) {
        if (op->getOpType() != OpType::Matmul)
            return false;
        if (op->getPredecessors().size() > 0)
            return false;
        if (op->getSuccessors().size() > 0)
            return false;
    }
    auto op0 = as<MatmulObj>(in_graph->getOperators()[0]);
    auto op1 = as<MatmulObj>(in_graph->getOperators()[1]);
    auto args0 = op0->getBMNKTransAB();
    auto args1 = op1->getBMNKTransAB();
    return args0 == args1;
}

} // namespace infini
