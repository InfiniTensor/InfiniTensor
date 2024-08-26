#include "core/graph.h"
#include "core/runtime.h"
#include "operators/ascend_quant.h"
#include "test.h"

namespace infini {
TEST(AscendQuant, ShapeInference) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(cpuRuntime);
        Tensor inputNpu = g->addTensor(Shape{1, 3}, DataType::Float32);
        auto op = g->addOp<AscendQuantObj>(inputNpu, nullptr,
                                           vector<float>{1.0, 2.0, 3.0},
                                           vector<float>{0.3, 0.5, 0.7});
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 3}));
    }
}

} // namespace infini
