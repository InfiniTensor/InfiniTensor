#pragma once

#include "common.h"
#include "graph.h"
#include "mutator.h"

namespace infini {
class SearchEngine {
  private:
    Runtime runtimeExec;
    Ref<Mutator> mutator;
    std::function<bool(const Graph &, const Graph &)> graphTimeComparer;

  public:
    SearchEngine(Runtime runtime, Ref<Mutator> mutator);
    ~SearchEngine() {}
    int searchFilter = 0;
    bool chooseBestMutation = true;

  private: // Configurations
    size_t partitionThreshold =
        3;                  // cut nodes whose #in + #out >= partitionThreshold
    size_t GRAPH_SIZE = 16; // num of best graphs.

  public:
    struct GroupEdge {
        int v, next;
        GroupEdge() = delete;
    };

    // struct Candidate { // a graph with perf
    //     Graph graph;
    //     double perf = INFINITY;
    // };
    struct MetaGraphObj { // a graph of subgraphs, for searching.
        struct Node {
            Graph graph;
            std::vector<int> suc;
            std::vector<int> pre;
            int type, cnt;
        };
        std::vector<Node> nodes;
    };
    using MetaGraph = Ref<MetaGraphObj>;

    Graph run(const Graph graph);                  // entrance to search engine.
    std::vector<Graph> search(const Graph &graph); // search for a partition.

  private:
    std::vector<Graph> partitionGraph(const Graph graph);
    MetaGraph buildMetaGraphWithGraph(const Graph graph);
    MetaGraph buildMetaGraphWithPlan(const MetaGraph metaGraph,
                                     const std::vector<int> &plan);
    // search horizontal merges
    std::vector<MetaGraph> searchMerge(MetaGraph &metaGraph);
    void searchMergeDfs(MetaGraph &metaGraph, std::vector<int> &plan,
                        std::vector<int> &frontier,
                        std::vector<std::vector<int>> &plans,
                        std::unordered_set<uint64_t> &planSet);
    std::vector<Graph> searchMutation(const MetaGraph &metaGraph);

    void printMetaGraph(MetaGraph metaGraph);
    /**
     * @brief Check whether a multi-brach graph can be merged into a single
     * branch.
     */
    bool isMultiBranchMergable(const Graph graph);
    Graph fuseVertically(const Graph &graph);

    double getEstimatedGraphPerf(Graph graph);
};

} // namespace infini
