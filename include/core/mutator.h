#pragma once
#include "core/graph.h"

namespace infini {

class Mutator {
  public:
    Mutator(){};
    virtual ~Mutator(){};

    virtual vector<Graph> run(const Graph &in_graph) = 0;
};

} // namespace infini