#include "core/graph_factory.h"
#include "test.h"

namespace infini {

TEST(GraphFactory, build_and_run) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    GraphFactory gf = make_ref<GraphFactoryObj>(runtime);
}

} // namespace infini