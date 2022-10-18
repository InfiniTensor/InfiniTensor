#pragma once

#include "common.h"
#include "graph.h"
#include "mutator.h"
#include "operator.h"

#include <unordered_map>

namespace infini {
class SearchEngine {
  private:
    std::vector<Graph> partitionGraph(const Graph graph);

  public:
    SearchEngine() = default;
    // SearchEngine is singleton
    SearchEngine(SearchEngine &other) = delete;
    SearchEngine &operator=(SearchEngine const &) = delete;

  private: // Configurations
    int MUTATION_DEPTH;
    int MUTATION_SIZE;
    int partitionThreshold; // cut nodes whose #in + #out >= partitionThreshold
    int GRAPH_SIZE;
    bool enableMetagraphMerging; // searchDfs
    bool enableVerification;     // compare the outputs of replacer and replacee

  private: // Composed objects
    std::shared_ptr<Mutator> mutationEngine;
    // std::shared_ptr<TransEliminator> eliminateEngine;
    std::unordered_map<uint64_t, std::vector<std::shared_ptr<Graph>>>
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
        std::shared_ptr<Graph> graph;
        double perf;
        Candidate() {
            graph = nullptr;
            perf = INFINITY;
        }
        Candidate(std::shared_ptr<Graph> graph_, double perf_) {
            graph = graph_;
            perf = perf_;
        }
        static bool cmp(const Candidate &a, const Candidate &b);
    };
    struct MetaGraph {
        struct Node {
            std::shared_ptr<Graph> graph;
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

    std::vector<Graph> run(const Graph graph);
    int search(const std::shared_ptr<Graph> &graph,
               std::vector<std::shared_ptr<Graph>> &bestGraphs);
    // Split a Graph on Non-linear OPs into a MetaGraph
    int split(const std::shared_ptr<Graph> &graph,
              std::shared_ptr<MetaGraph> &metaGraph);
    // Enumerate possible merges of OPs among Metagraph.nodes into new
    // Metagraphs
    int searchDfs(const std::shared_ptr<MetaGraph> &metaGraph,
                  std::vector<std::shared_ptr<MetaGraph>> &metaGraphs);
    int searchDfs(const std::shared_ptr<MetaGraph> &metaGraph,
                  std::vector<int> &frontier, std::vector<int> &f,
                  std::vector<std::vector<int>> &candidates,
                  std::unordered_set<uint64_t> &candidateSet);
    int searchBfs(const std::shared_ptr<MetaGraph> &metaGraph,
                  std::vector<Candidate> &candidates);

    int isMergeable(const std::shared_ptr<Graph> &graph);
    int isMutatable(const std::shared_ptr<Graph> &graph);
    int isSpecialMutation(Operator *, int depth);
    double getPerf(const std::shared_ptr<Graph> &graph, bool profiling = false,
                   bool withPenalty = true);
    double getMaxPerf(const std::shared_ptr<Graph> &graph,
                      bool profiling = false, bool withPenalty = true);
    double getTransPerf(const std::shared_ptr<Graph> &graph);
    int getMutation(std::shared_ptr<Graph> &graph,
                    std::vector<std::shared_ptr<Graph>> &mutatedGraphs);
    int getSingleMutation(std::shared_ptr<Graph> &graph,
                          std::vector<std::shared_ptr<Graph>> &candidates);
    uint64_t getMutationHash(const Operator *op);

    // Partition a graph into disjoint Graphs
    std::vector<std::shared_ptr<Graph>>
        // Fuse activations
        std::shared_ptr<Graph> fuse(const std::shared_ptr<Graph> &graph);
    // Remove redundant transpositions
    std::shared_ptr<Graph> strip(const std::shared_ptr<Graph> &graph);
    int stripDfs(Operator *op, std::unordered_map<int, int> &f, int flag);

    Operator *FuseMemBoundChain(std::vector<Operator *> chainOps);
};
// nnet::Expr transposeOpToExpression(TransposeOp *transposeOp);
} // namespace infini