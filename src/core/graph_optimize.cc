#include "core/graph_optimize.h"
#include "operators/unary.h"

namespace infini {

bool fuseSigmoidMulToSilu(GraphObj *graph, const Operator &sigmoidOp) {
    // isCuda() covers CUDA, MetaX and CoreX runtimes.
    if (!graph->getRuntime()->isCuda())
        return false;

    if (sigmoidOp->getOpType() != OpType::Sigmoid)
        return false;

    const auto x = sigmoidOp->getInputs(0);
    const auto sigmoidOut = sigmoidOp->getOutput();
    const auto targets = sigmoidOut->getTargets();
    // An exported Sigmoid result must remain available to the caller.
    if (!sigmoidOut->isOthers() || targets.size() != 1)
        return false;
    const auto mulOp = targets.front();
    if (mulOp->getOpType() != OpType::Mul)
        return false;
    const auto &inputs = mulOp->getInputs();
    if (inputs.size() != 2 || !((inputs[0] == x && inputs[1] == sigmoidOut) ||
                                (inputs[1] == x && inputs[0] == sigmoidOut)))
        return false;

    const auto source = x->getSource();
    if (source && (source->getOpType() == OpType::Split ||
                   source->getOutputs().size() > 1))
        return false;

    const auto mulOut = mulOp->getOutput();
    if (mulOut->getDims() != x->getDims() ||
        !(mulOut->getDType() == x->getDType()))
        return false;

    // Keep the final output's production point at the original Mul position.
    graph->replaceOperator(mulOp, make_ref<SiluObj>(nullptr, x, mulOut));
    graph->deleteConnection(x, sigmoidOp);
    graph->removeOperator(sigmoidOp);
    graph->removeTensor(sigmoidOut);
    return true;
}

bool optimizeGraph(GraphObj *graph, int maxFusions) {
    int count = 0;
    // Erasing Sigmoid/Mul must not invalidate traversal or skip candidates.
    const auto candidates = graph->getOperators();
    for (const auto &op : candidates) {
        if (maxFusions >= 0 && count >= maxFusions)
            break;
        if (fuseSigmoidMulToSilu(graph, op))
            ++count;
    }
    if (count)
        IT_ASSERT(graph->topo_sort(), "Fusion produced an invalid graph");
    return count != 0;
}

} // namespace infini
