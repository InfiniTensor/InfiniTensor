#include "core/graph.h"
#include "core/runtime.h"
#include "operators/rope.h"

#include "test.h"

namespace infini {

TEST(RoPE, InferOutputDataType) {
    auto runtime = NativeCpuRuntimeObj::getInstance();

    {
        Graph graph = make_ref<GraphObj>(runtime);
        auto positions = graph->addTensor({1, 2}, DataType::Int32);
        auto input = graph->addTensor({1, 2, 128}, DataType::Float32);
        auto rope = graph->addOp<RoPEObj>(positions, input, nullptr);

        EXPECT_EQ(rope->getOutput()->getDims(), (Shape{1, 2, 128}));
        EXPECT_EQ(rope->getOutput()->getDType(), DataType::Float32);
    }

    {
        Graph graph = make_ref<GraphObj>(runtime);
        auto positions = graph->addTensor({1, 2}, DataType::Int64);
        auto input = graph->addTensor({1, 2, 128}, DataType::Float16);
        auto rope = graph->addOp<RoPEObj>(positions, input, nullptr);

        EXPECT_EQ(rope->getOutput()->getDims(), (Shape{1, 2, 128}));
        EXPECT_EQ(rope->getOutput()->getDType(), DataType::Float16);
    }
}

TEST(RoPE, PreserveExplicitOutput) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph graph = make_ref<GraphObj>(runtime);
    auto positions = graph->addTensor({1, 2}, DataType::Int32);
    auto input = graph->addTensor({1, 2, 128}, DataType::Float32);
    auto output = graph->addTensor({1, 2, 128}, DataType::Float32);
    auto rope = graph->addOpWithOutputs<RoPEObj>(positions, input, output);

    EXPECT_EQ(rope->getOutput(), output);
    EXPECT_EQ(rope->getOutput()->getDims(), (Shape{1, 2, 128}));
    EXPECT_EQ(rope->getOutput()->getDType(), DataType::Float32);
}

} // namespace infini
