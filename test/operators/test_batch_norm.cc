#include "core/graph.h"
#include "core/runtime.h"
#include "operators/batch_norm.h"
#include "test.h"

namespace infini {
TEST(BatchNorm, ShapeInference) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(cpuRuntime);
        Tensor i = g->addTensor({1, 3, 2, 2}, DataType::UInt32);
        Tensor mean = g->addTensor({3}, DataType::Float32);
        Tensor var = g->addTensor({3}, DataType::Float32);
        Tensor scaler = g->addTensor({3}, DataType::Float32);
        Tensor bias = g->addTensor({3}, DataType::Float32);
        auto op = g->addOp<BatchNormObj>(i, nullptr, mean, var, scaler, bias,
                                         0.9, 1e-5);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 3, 2, 2}));
    }
}
} // namespace infini
