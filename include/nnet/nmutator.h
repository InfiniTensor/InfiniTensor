#pragma once
#include "core/mutator.h"
#include "nnet/expr.h"

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

namespace infini {

class NMutator : public Mutator {
  public:
    enum class Mode { Normal, ToNaiveMembound, RuleBased };

  private:
    // Suffix -N: NNet objects.
    // Suffix -T: tpm objects.
    // Map: NNet tensors -> tpm tensor.
    std::map<std::string, Tensor> inputsNameNToTensorT;
    Mode mode;
    const double bandwidth = double(200) * 1024 * 1024 * 1024;
    // If in RuleBased mode, use derivationRules in derivator
    const std::vector<int> derivationRules;

  public:
    NMutator(Mode mode = Mode::Normal);
    NMutator(Mode mode, const std::vector<int> &derivationRules);
    ~NMutator();

    vector<Graph> run(const Graph &in_graph) override;
    void setToNaiveMembound();

    void setMaxDepth(int _maxDepth) { maxDepth = _maxDepth; }
    long long cntStates = 0;
    long long cntCandidates = 0;

  private:
    int maxDepth = 8;
    nnet::Expr opToExpression(Operator op);
    void runSingleOp(Graph in_graph, std::vector<Graph> &out_graphs);

    /**
     * @brief Test helper. Converting a single OP to Membound Op for
     * corretness check.
     */
    void runSingleOpToNaiveMembound(Graph in_graph,
                                    std::vector<Graph> &out_graphs);
    void runMultipleOps(Graph in_graph, std::vector<Graph> &out_graphs);
    Graph expressionToGraph(nnet::Expr expr, Graph in_graph);
    double memboundTime(ssize_t cnt);
    double memboundTime(const Shape &dims);

    // TODO: recover these rules
    // Graph fuseHetConv(nnet::Expr expr, Graph in_graph);
    // Graph transformTConv1x1(Operator op);
    // Graph transformTConv3x3(Operator op);
    // Graph transformDialtedConv(Operator op);
    // Graph transformConv1x1(Operator op);
    // Graph transformConv1xk(Operator op);
};

} // namespace infini
