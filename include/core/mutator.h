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
        : candidatesLimit(candidatesLimit), runtime(runtime){};
    virtual ~Mutator(){};
    bool hasTunedKernel = false;

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

    /// @brief Fuse memory bound operators.
    /// @return The graph after fusion. Return `nullptr` if fails.
    virtual Graph fuseVertically(const Graph &inputGraph) { IT_TODO_HALT(); }

    /// @brief Eliminate transpose and reshape.
    /// @return The graph after elimination. Return `nullptr` if fails.
    virtual Graph eliminateVertically(const Graph &in_graph) { IT_TODO_HALT(); }
};

} // namespace infini
