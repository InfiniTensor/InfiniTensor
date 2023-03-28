#include "core/graph.h"
#include "pass.h"

namespace infini {
class PassManager {
  public:
    PassManager() {}

    Graph run(const Graph graph);

    bool addPass(Ref<Partition> partition, Ref<Transformation> transformation);

    bool addPass(Ref<Transformation> transformation);

    string toString();

  private:
    vector<Ref<Pass>> passes;
};
} // namespace infini