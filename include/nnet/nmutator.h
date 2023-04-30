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
    bool searchFilter = false;

  public:
    NMutator(Mode mode = Mode::Normal,
             Runtime runtime = NativeCpuRuntimeObj::getInstance());
    NMutator(Mode mode, const std::vector<int> &derivationRules,
             Runtime runtime = NativeCpuRuntimeObj::getInstance());
    ~NMutator();

    vector<Graph> run(const Graph &in_graph) override;
    Graph fuseVertically(const Graph &in_graph) override;
    Graph eliminateVertically(const Graph &in_graph) override;
    bool isMultiBranchMergable(const Graph &in_graph) override;

    void setToNaiveMembound();
    void setMaxDepth(int _maxDepth) {
        maxDepth = _maxDepth;
        searchFilter = true;
    }
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
    Graph transformConvtransposed1x1(Operator _op);
    // Graph transformConvtransposed(Operator op);
    vector<Graph> transformConv1x1(Operator op);
    vector<Graph> transformConv3x3ONNX(Operator op);
    Graph transformG2bmm(Operator op);
    Graph transformGbmm(Operator op);
    Graph transformDialtedConv(Operator _op);
    vector<Graph> transformConv1xk(Operator op);
    // Graph transformConv1xk(Operator op);
    Graph transformConvToGEMMReduce(Operator _op);
    Graph transformConvTranposeToGEMMReduce(Operator _op);

    Tensor splitTransposeMerge(Graph g, Tensor A, int dim, int chunkSize,
                               Tensor output = nullptr);

    /// @brief Construct a new graph with a chain of operators. Use the output
    /// from the previous operator as the input of the next operator. While
    /// constructing, the input and output tensors from inputGraph are used as
    /// new constructed graph.
    /// @param op The operator chain. It can have wrong input/output shapes.
    /// @return
    Graph constructGraphByOperatorChain(vector<Operator> ops, Graph inputGraph);
};

} // namespace infini
