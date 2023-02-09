#pragma once

#include "common.h"
#include "graph.h"
#include "mutator.h"

#include <unordered_map>

namespace infini {
class SearchEngine {
  private:
    Runtime runtimeExec, runtimeVerification;
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
        GroupEdge(int v_, int next_) {
            v = v_;
            next = next_;
        }
    };
    struct Candidate { // a graph with perf
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
    // TODO: move to cpp
    bool isMergeable(const Graph graph);
};
} // namespace infini