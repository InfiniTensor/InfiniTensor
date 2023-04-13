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
    std::function<bool(const Graph &, const Graph &)> graphTimeComparer;

  public:
    SearchEngine(Runtime runtime, Ref<Mutator> mutator)
        : runtimeExec(runtime), mutator(mutator) {
        // Compare graph with estimated time
        graphTimeComparer = [this](const Graph &a, const Graph &b) -> bool {
            return getEstimatedGraphPerf(a) < getEstimatedGraphPerf(b);
        };
    }
    ~SearchEngine() {}

  private: // Configurations
    size_t partitionThreshold =
        3;                  // cut nodes whose #in + #out >= partitionThreshold
    size_t GRAPH_SIZE = 16; // num of best graphs.

  public:
    struct GroupEdge {
        int v, next;
        GroupEdge() = delete;
    };

    struct Candidate { // a graph with perf
        std::shared_ptr<Graph> graph;
        double perf = INFINITY;
    };
    struct MetaGraph { // a graph of subgraphs, for searching.
        struct Node {
            Graph graph;
            std::vector<int> suc;
            std::vector<int> pre;
            int type, cnt;
        };
        std::vector<Node> nodes;
    };

    Graph run(const Graph graph);                  // entrance to search engine.
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

    double getEstimatedGraphPerf(Graph graph) {
        return runtimeExec->getPerfTime(graph, false, true);
    }
};
} // namespace infini
