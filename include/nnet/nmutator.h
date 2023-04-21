#pragma once
#include "core/mutator.h"
#include "nnet/expr.h"

namespace infini {

class NMutator : public Mutator {
  public:
    enum class Mode { Normal, ToNaiveMembound, RuleBased };
    using NameNToTensorT = map<string, Tensor>;

  private:
    // Suffix -N: NNet objects.
    // Suffix -T: tpm objects.
    // Map: NNet tensors -> tpm tensor.
    NameNToTensorT inputsNameNToTensorT;
    Mode mode;
    const double bandwidth = double(200) * 1024 * 1024 * 1024;
    // If in RuleBased mode, use derivationRules in derivator
    const std::vector<int> derivationRules;

  public:
    NMutator(Mode mode = Mode::Normal);
    NMutator(Mode mode, const std::vector<int> &derivationRules);
    ~NMutator();

    vector<Graph> run(const Graph &in_graph) override;
    Graph fuseVertically(const Graph &in_graph) override;

    void setToNaiveMembound();
    void setMaxDepth(int _maxDepth) { maxDepth = _maxDepth; }
    long long cntStates = 0;
    long long cntCandidates = 0;

  private:
    int maxDepth = 8;
    nnet::Expr opToExpression(Operator op);
    /// @brief
    /// @param op
    /// @return pair<Expr, map from NNet tensor names to InfiniTensor tensors>
    static pair<nnet::Expr, NameNToTensorT> extractOp(Operator op);
    static pair<nnet::Expr, NMutator::NameNToTensorT>
    generateUnaryExpr(const Operator &op);
    static pair<nnet::Expr, vector<nnet::Tensor>> generateRevert(Tensor in);

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
    Graph transformConvtransposed1x1(Operator _op);
    // Graph transformConvtransposed(Operator op);
    Graph transformDialtedConv(Operator _op);
    // Graph transformConv1x1(Operator op);
    // Graph transformConv1xk(Operator op);
};

} // namespace infini
