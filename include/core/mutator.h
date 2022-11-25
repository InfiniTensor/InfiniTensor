#pragma once
#include "core/graph.h"

namespace infini {

class Mutator {
  private:
    int candidatesLimit;
    // // Statistical data
    // int numTotalCandidates;

  public:
    Mutator(int candidatesLimit) : candidatesLimit(candidatesLimit){};
    virtual ~Mutator(){};

    virtual vector<Graph> run(const Graph &in_graph) = 0;
    virtual vector<Graph> fusion(const Graph &in_graph) { IT_ASSERT(false); };
    virtual bool isFusible() { IT_ASSERT(false); }
};

} // namespace infini
