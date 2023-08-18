#include "core/graph.h"
#include "core/runtime.h"
#include "operators/all_gather.h"
#include "test.h"

namespace infini {
TEST(AllGather, ShapeTypeInfer) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    int world_size = 8;
    {
        Shape shape = {1, 3, 2, 4};
        Graph g = make_ref<GraphObj>(runtime);
        Tensor input = g->addTensor(shape, DataType::Float32);
        auto op = g->addOp<AllGatherObj>(input, std::nullopt, world_size);
        EXPECT_EQ(op->getOpType(), OpType::AllGather);
        EXPECT_EQ(op->numOutputs(), world_size);
        for (int i = 0; i < world_size; ++i) {
            EXPECT_EQ(op->getOutput(i)->getDims(), shape);
            EXPECT_EQ(op->getOutput(i)->getDType(), DataType::Float32);
        }
    }
}
} // namespace infini