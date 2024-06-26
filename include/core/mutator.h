#pragma once
#include "core/graph.h"

namespace infini {

class Mutator {
  private:
    int candidatesLimit;
    // // Statistical data
    // int numTotalCandidates;
  protected:
    Runtime runtime;

  public:
    Mutator(int candidatesLimit,
            Runtime runtime = NativeCpuRuntimeObj::getInstance())
        : candidatesLimit(candidatesLimit), runtime(runtime) {};
    virtual ~Mutator() {};

    virtual vector<Graph> run(const Graph &in_graph) = 0;
    /**
     * @brief Merge a multi-branch graph into single branch graphs
     *
     * @param in_graph
     * @return vector<Graph> Transformed graphs except the orignal one.
     */
    virtual vector<Graph> mergeMultiBranch(const Graph &in_graph) {
        IT_TODO_HALT();
    }
    virtual bool isMultiBranchMergable(const Graph &in_graph) {
        IT_TODO_HALT();
    }
};

} // namespace infini
