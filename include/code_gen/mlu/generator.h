#ifndef GENERATOR_H
#define GENERATOR_H

#include "mutator.h"
#include "reciprocity.h"
#include <unordered_set>

#define EQOPT                                                                  \
    if (!enable_eq_opt) {                                                      \
        return;                                                                \
    }
#define NEQOPT                                                                 \
    if (!enable_non_eq_opt) {                                                  \
        return;                                                                \
    }

namespace tpm {

class Reciprocity;

class Generator : public Mutator {
    float equal_threshold;
    size_t num_valid_tensors, num_total_tensors;
    int max_depth;
    SubGraph *searchingGraph;
    size_t max_num_elements;
    size_t num_reserve_ops;
    size_t group_size;
    std::vector<OpVec> all_ops;
    OpVec oplist;

    // enable reciprocity pruning
    bool prune_reciprocity;
    std::shared_ptr<Reciprocity> reciprocity;

    std::vector<std::vector<std::pair<Dim, VType>>> computingPos;

    OpVec computation_ops;

    // box verification
    bool enable_box_verification = false;

    std::map<uint64_t, std::vector<std::shared_ptr<SubGraph>>> mutationCache;

    bool enable_eq_opt, enable_non_eq_opt;

  public:
    Generator(bool prune_reciprocity = true);
    ~Generator() {
        if (searchingGraph != nullptr)
            delete searchingGraph;
    }
    void run(SubGraph *in_graph, std::vector<SubGraph *> &out_graphs,
             int mdepth = -1,
             // Do not use reference since candidates_ops may be modified in
             // this function
             std::vector<std::shared_ptr<Operator>> candidate_ops = {},
             float threshold = 0.7);
    void dfs(int depth, SubGraph *in_graph, SubGraph *cur_graph,
             std::vector<SubGraph *> &out_graphs,
             std::unordered_set<uint64_t> &visited);
    bool is_a_mutant(const SubGraph *mutant_graph, const SubGraph *input_graph,
                     bool full_computing = true);
    bool approx_equal(Tensor *a, Tensor *b);
    bool approx_equal(const SubGraph *mutant_graph, size_t midx,
                      const SubGraph *input_graph, size_t iidx);
    bool approx_equal_splitting_points(const SubGraph *mutant_graph,
                                       size_t midx, const SubGraph *input_graph,
                                       size_t iidx);

    void runForGroupConv(SubGraph *in_graph,
                         std::vector<SubGraph *> &out_graphs);
    SGType statGraph(SubGraph *sg);
    uint64_t computeHashForSingleComputeOp(const Operator *op);

  private:
    // // find reciprocities among given ops
    // void search_reciprocity(OpVec &ops);
    // add and remove op to oplist
    bool pushBackOp(Operator *op);
    bool popBackOp();
    // add and remove tensor
    Tensor *newTensor();
    bool popBackTensor(Operator *op = nullptr);
    void reserveTensors(size_t size);
    // pruning if there is dependence among computeOps
    bool have_computeOp_ancestor(Tensor *tensor);
    // pruning if an op with the same inputs exist
    bool have_same_op(Operator *op);

    void resetGraph(const SubGraph *in_graph);

    void addCandidateOpsForConv1x1(
        std::vector<std::shared_ptr<Operator>> &candidate_ops, SubGraph *sg);
    void addCandidateOpsForNormalConv(
        std::vector<std::shared_ptr<Operator>> &candidate_ops, SubGraph *sg);
    void addCandidateOpsForNormalOddConv(
        std::vector<std::shared_ptr<Operator>> &candidate_ops, SubGraph *sg);
    void addCandidateOpsForDilatedConv(
        std::vector<std::shared_ptr<Operator>> &candidate_ops, SubGraph *sg);
    void addCandidateOpsForTransKernelConv(
        std::vector<std::shared_ptr<Operator>> &candidate_ops, SubGraph *sg);
    void addCandidateOpsForGroupConv(
        std::vector<std::shared_ptr<Operator>> &candidate_ops, SubGraph *sg);
    void addCandidateOpsForTransposeGroupConv(
        std::vector<std::shared_ptr<Operator>> &candidate_ops, SubGraph *sg);
    void addCandidateOpsForNormalMatmul(
        std::vector<std::shared_ptr<Operator>> &candidate_ops, SubGraph *sg);
    void addCandidateOpsForBatchMatmul(
        std::vector<std::shared_ptr<Operator>> &candidate_ops, SubGraph *sg);
    void addPreprocessForConv1x1(SubGraph *sg);
    void addPreprocessForGroupConvGCD(SubGraph *sg);
    void addPreprocessForGroupConvMAX(SubGraph *sg);
    void addPreprocessForGroupConvOneInput(SubGraph *sg);
    void addPreprocessForPadSlice(SubGraph *sg);
    void addPreprocessForTransKernel(SubGraph *sg);
    void addPreprocessForBatchMatmul(SubGraph *sg);
    // TODO: merge rs and sr together
    void addPreprocessForTransposeGroupConvRS(SubGraph *sg);
    void addPreprocessForTransposeGroupConvSR(SubGraph *sg);
    uint64_t computeHashForSingleConv(Operator *op);
    void addToCache(SubGraph *sg, std::vector<SubGraph *> &out_graphs);
    void markTransType(SubGraph *inputGraph, SubGraph *outputGraph);
    void splitGroupConv(SubGraph *sg, std::vector<SubGraph *> &out_graphs);
    bool validDepth(SubGraph *sg);
};
} // end of namespace tpm
#endif
