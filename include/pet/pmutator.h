#pragma once
#include "core/mutator.h"

namespace infini {

class PMutator : public Mutator {
  public:
    enum class Mode { Normal, RuleBased };

  private:
    Mode mode;
    const double bandwidth = double(200) * 1024 * 1024 * 1024;
    // If in RuleBased mode, use derivationRules in derivator
    bool searchFilter = false;

  public:
    PMutator(Mode mode = Mode::Normal,
             Runtime runtime = NativeCpuRuntimeObj::getInstance());
    ~PMutator();

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

    void runSingleOp(Graph in_graph, std::vector<Graph> &out_graphs);

    void runMultipleOps(Graph in_graph, std::vector<Graph> &out_graphs);
    double memboundTime(ssize_t cnt);
    double memboundTime(const Shape &dims);

    // TODO: recover these rules
    vector<Graph> transformConv1x1(Operator op);
    Graph transformDialtedConv(Operator _op);
    // Graph transformation for conv between NHW
    Graph transformConvW2N(Operator _op);
    Graph transformConvH2N(Operator _op);
    Graph transformConvH2W(Operator _op);
    Graph transformConvW2H(Operator _op);
    Graph transformConvN2H(Operator _op);
    Graph transformConvN2W(Operator _op);

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
