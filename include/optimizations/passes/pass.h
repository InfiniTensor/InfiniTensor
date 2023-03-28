#include "core/graph.h"
#include "optimizations/partitions/partition.h"

namespace infini {
class Pass {
  public:
    Pass() {}

    virtual vector<Graph> run(const Graph graph) = 0;
};
} // namespace infini