#include "core/graph.h"
#include "core/runtime.h"
#include "operators/ascend_dequant.h"
#include "test.h"

namespace infini {
TEST(AscendDequant, ShapeInference) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(cpuRuntime);
        Tensor inputNpu = g->addTensor(Shape{1, 3}, DataType::Int8);
        auto op = g->addOp<AscendDequantObj>(inputNpu, nullptr,
                                             vector<float>{1.0, 2.0, 3.0},
                                             vector<float>{0.3, 0.5, 0.7});
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 3}));
    }
}

} // namespace infini
