#include "core/graph.h"
#include "pass.h"

namespace infini {
class PassManager {
  public:
    PassManager() {}

    Graph run(const Graph graph);

    bool addPass(Ref<Partition>, Ref<Transformation>, Ref<Rating>);

    string toString();

  private:
    vector<Ref<Pass>> passes;
};
} // namespace infini
