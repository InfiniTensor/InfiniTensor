#pragma once

#include "common.h"
#include "generator.h"
#include "graph.h"
#include "mutator.h"
#include "operator.h"
#include "trans_eliminator.h"
#include <unordered_map>

namespace tpm {
class SearchEngine {
  private:
    int MUTATION_DEPTH = 5;
    int MUTATION_SIZE = 5;
    int partitionThreshold =
        3; // cut nodes whose #in + #out >= partitionThreshold
    int GRAPH_SIZE = 5;
    std::shared_ptr<PerfEngine> perfEngine;
    std::shared_ptr<Mutator> mutationEngine;
    std::shared_ptr<TransEliminator> eliminateEngine;
    std::unordered_map<uint64_t, std::vector<std::shared_ptr<SubGraph>>>
        mutationArchive;

  public:
    std::shared_ptr<Mutator> getMutationEngine() { return mutationEngine; };
    struct GroupEdge {
        int v, next;
        GroupEdge(int v_, int next_) {
            v = v_;
            next = next_;
        }
    };
    struct Candidate {
        std::shared_ptr<SubGraph> graph;
        double perf;
        Candidate() {
            graph = nullptr;
            perf = INFINITY;
        }
        Candidate(std::shared_ptr<SubGraph> graph_, double perf_) {
            graph = graph_;
            perf = perf_;
        }
        static bool cmp(const Candidate &a, const Candidate &b);
    };
    struct MetaGraph {
        struct Node {
            std::shared_ptr<SubGraph> graph;
            std::vector<int> suc;
            std::vector<int> pre;
            int type, cnt;
        };
        MetaGraph() {}
        ~MetaGraph() {}
        int print();
        std::vector<Node> nodes;
    };
    SearchEngine(const std::shared_ptr<Mutator> &mutationEngine);
    ~SearchEngine();

    int run(const std::shared_ptr<SubGraph> &graph,
            std::shared_ptr<SubGraph> &bestGraph);
    int search(const std::shared_ptr<SubGraph> &graph,
               std::vector<std::shared_ptr<SubGraph>> &bestGraphs);
    int split(const std::shared_ptr<SubGraph> &graph,
              std::shared_ptr<MetaGraph> &metaGraph);
    int searchDfs(const std::shared_ptr<MetaGraph> &metaGraph,
                  std::vector<std::shared_ptr<MetaGraph>> &metaGraphs);
    int searchDfs(const std::shared_ptr<MetaGraph> &metaGraph,
                  std::vector<int> &frontier, std::vector<int> &f,
                  std::vector<std::vector<int>> &candidates,
                  std::unordered_set<uint64_t> &candidateSet);
    int searchBfs(const std::shared_ptr<MetaGraph> &metaGraph,
                  std::vector<Candidate> &candidates);

    int isMergeable(const std::shared_ptr<SubGraph> &graph);
    int isMutatable(const std::shared_ptr<SubGraph> &graph);
    int isSpecialMutation(Operator *, int depth);
    double getPerf(const std::shared_ptr<SubGraph> &graph,
                   bool profiling = false, bool withPenalty = true);
    double getMaxPerf(const std::shared_ptr<SubGraph> &graph,
                      bool profiling = false, bool withPenalty = true);
    double getTransPerf(const std::shared_ptr<SubGraph> &graph);
    int getMutation(std::shared_ptr<SubGraph> &graph,
                    std::vector<std::shared_ptr<SubGraph>> &mutatedGraphs);
    int getSingleMutation(std::shared_ptr<SubGraph> &graph,
                          std::vector<std::shared_ptr<SubGraph>> &candidates);
    uint64_t getMutationHash(const Operator *op);

    // Partition a graph into disjoint subgraphs
    std::vector<std::shared_ptr<SubGraph>>
    partition(const std::shared_ptr<SubGraph> &graph);
    // Fuse activations
    std::shared_ptr<SubGraph> fuse(const std::shared_ptr<SubGraph> &graph);
    // Remove redundant transpositions
    std::shared_ptr<SubGraph> strip(const std::shared_ptr<SubGraph> &graph);
    int stripDfs(Operator *op, std::unordered_map<int, int> &f, int flag);

    std::shared_ptr<PerfEngine> exportPerfEngine();

    Operator *FuseMemBoundChain(std::vector<Operator *> chainOps);
};
nnet::Expr transposeOpToExpression(TransposeOp *transposeOp);
} // namespace tpm
