#include "core/common.h"
#include "core/graph.h"
#include "core/runtime.h"

namespace infini {
class Transformation {
  public:
    Transformation() {}

    virtual vector<Graph> run(const Graph graph) { return {Graph(graph)}; };
};
} // namespace infini
