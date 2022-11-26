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
    Mutator(int candidatesLimit, Runtime runtime = CpuRuntimeObj::getInstance())
        : candidatesLimit(candidatesLimit), runtime(runtime){};
    virtual ~Mutator(){};

    virtual vector<Graph> run(const Graph &in_graph) = 0;
    virtual vector<Graph> fusion(const Graph &in_graph) { IT_TODO_HALT(); }
    virtual bool isFusible(const Graph &in_graph) { IT_TODO_HALT(); }
};

} // namespace infini
