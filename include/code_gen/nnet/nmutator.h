#ifndef NMUTATOR_H
#define NMUTATOR_H

#include "code_gen/mutator.h"
#include "code_gen/nnet/expr.h"

namespace tpm {

class NMutator : public Mutator {
  private:
    // Suffix -N: NNet objects.
    // Suffix -T: tpm objects.
    // Map: NNet tensors -> tpm tensor.
    std::map<std::string, Tensor *> inputsNameNToTensorT;
    enum class Mode { Normal, ToNaiveMembound } mode = Mode::Normal;
    const double bandwidth = double(200) * 1024 * 1024 * 1024;

  public:
    NMutator();
    ~NMutator();

    void run(SubGraph *in_graph, std::vector<SubGraph *> &out_graphs,
             int mdepth = -1,
             std::vector<std::shared_ptr<Operator>> candidate_ops = {},
             float threshold = 0.7);
    void setToNaiveMembound();

    SGType statGraph(SubGraph *sg);

    uint64_t computeHashForSingleComputeOp(const Operator *op);

    void setMaxDepth(int _maxDepth) { maxDepth = _maxDepth; }
    long long cntStates = 0;
    long long cntCandidates = 0;

  private:
    int maxDepth = 8;
    nnet::Expr opToExpression(Operator *op);
    void runSingleOp(SubGraph *in_graph, std::vector<SubGraph *> &out_graphs);

    /**
     * @brief Test helper. Converting a single OP to Membound Op for corretness
     * check.
     */
    void runSingleOpToNaiveMembound(SubGraph *in_graph,
                                    std::vector<SubGraph *> &out_graphs);
    void runMultipleOps(SubGraph *in_graph,
                        std::vector<SubGraph *> &out_graphs);
    tpm::SubGraph *expressionToGraph(nnet::Expr expr, SubGraph *in_graph);
    tpm::SubGraph *fuseHetConv(nnet::Expr expr, SubGraph *in_graph);
    double memboundTime(ssize_t cnt);
    double memboundTime(const Dim &dims);

    tpm::SubGraph *transformTConv1x1(Operator *op);
    SubGraph *transformTConv3x3(Operator *op);
    SubGraph *transformDialtedConv(Operator *op);
    SubGraph *transformConv1x1(Operator *op);
    SubGraph *transformConv1xk(Operator *op);
};

} // namespace tpm

#endif
