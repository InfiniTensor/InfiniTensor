#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/range.h"

#include "test.h"

namespace infini {

TEST(rangeFloat, run) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(runtime);
        float start = 1.0;
        float limit = 6.0;
        float delta = 1.0;

        auto op = g->addOp<RangeObj>(start, limit, delta, nullptr);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{5}));

    }

}

} // namespace infini
