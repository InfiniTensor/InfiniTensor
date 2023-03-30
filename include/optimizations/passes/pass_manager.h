#include "core/graph.h"
#include "pass.h"

namespace infini {
class PassManager {
  public:
    PassManager() {}

    Graph run(Graph graph) {
        for (auto pass : passes)
            graph = pass->run(*graph);
        return graph;
    }

    bool addPass(std::unique_ptr<Partition> p,
                 std::unique_ptr<Transformation> tr,
                 std::unique_ptr<Rating> rating) {
        passes.emplace_back(std::move(p), std::move(tr), std::move(rating));
        return true;
    }

  private:
    vector<Ref<Pass>> passes;
};
} // namespace infini
