#pragma once

#include "common.h"
#include "graph.h"
#include "mutator.h"

#include <unordered_map>

namespace infini {
class SearchEngine {
  private:
    Runtime runtimeExec;
    Ref<Mutator> mutator;

  public:
    SearchEngine(Runtime _runtime, Ref<Mutator> _mutator) {
        runtimeExec = _runtime;
        mutator = _mutator;
    }
    ~SearchEngine() {}

  private: // Configurations
    size_t partitionThreshold =
        3;                  // cut nodes whose #in + #out >= partitionThreshold
    size_t GRAPH_SIZE = 16; // num of best graphs.

  private: // Composed objects
    std::shared_ptr<Mutator> mutationEngine;

  public:
    std::shared_ptr<Mutator> getMutationEngine() { return mutationEngine; };
    struct GroupEdge {
        int v, next;
        GroupEdge() = delete;
    };

    struct Candidate { // a graph with perf
        std::shared_ptr<Graph> graph;
        double perf = INFINITY;
    };
    class MetaGraph { // a graph of subgraphs, for searching.
      public:
        MetaGraph() {}
        ~MetaGraph() {}
        struct Node {
            Graph graph;
            std::vector<int> suc;
            std::vector<int> pre;
            int type, cnt;
        };
        std::vector<Node> nodes;
    };

    Graph run(const Graph graph);                  // entrance of search engine.
    std::vector<Graph> search(const Graph &graph); // search for a partition.

  private:
    std::vector<Graph> partitionGraph(const Graph graph);
    std::shared_ptr<MetaGraph> buildMetaGraphWithGraph(const Graph graph);
    std::shared_ptr<MetaGraph>
    buildMetaGraphWithPlan(const std::shared_ptr<MetaGraph> metaGraph,
                           const std::vector<int> &plan);
    // search horizontal merges
    std::vector<std::shared_ptr<MetaGraph>>
    searchMerge(std::shared_ptr<MetaGraph> &metaGraph);
    void searchMergeDfs(std::shared_ptr<MetaGraph> &metaGraph,
                        std::vector<int> &plan, std::vector<int> &frontier,
                        std::vector<std::vector<int>> &plans,
                        std::unordered_set<uint64_t> &planSet);
    std::vector<Graph>
    searchMutation(const std::shared_ptr<MetaGraph> &metaGraph);

    void printMetaGraph(Ref<SearchEngine::MetaGraph> metaGraph);
    /**
     * @brief Check whether a multi-brach graph can be merged into a single
     * branch.
     */
    bool isMultiBranchMergable(const Graph graph);
};
} // namespace infini
