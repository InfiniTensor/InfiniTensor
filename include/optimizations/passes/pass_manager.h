#include "core/graph.h"
#include "pass.h"

namespace infini {
class PassManager {
  public:
    PassManager() {}

    Graph run(const Graph graph);

    bool addPass(std::unique_ptr<Partition> p,
                 std::unique_ptr<Transformation> tr,
                 std::unique_ptr<Rating> rating) {
        passes.emplace_back(std::move(p), std::move(tr), std::move(rating));
    }

  private:
    vector<Ref<Pass>> passes;
};
} // namespace infini
