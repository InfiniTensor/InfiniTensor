#pragma once

#include "common.h"
#include "graph.h"
#include "mutator.h"
#include "operator.h"

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
    size_t
        partitionThreshold; // cut nodes whose #in + #out >= partitionThreshold
    size_t GRAPH_SIZE;
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

    Graph run(const Graph graph);
    std::vector<Graph> search(const Graph &graph);

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

  private:
    class MetaGraph {
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
    std::vector<Graph> partitionGraph(const Graph graph);
    std::shared_ptr<MetaGraph> buildMetaGraphWithGraph(const Graph graph);
    std::shared_ptr<MetaGraph>
    buildMetaGraphWithPlan(const std::shared_ptr<MetaGraph> metaGraph,
                           const std::vector<int> &plan) {
        IT_TODO_HALT();
        return nullptr;
    }
    // search merge
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