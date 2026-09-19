#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/conv.h"

#include "test.h"

namespace infini {

TEST(Conv, ShapeInference) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    // Padding modes
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i0 = g->addTensor({1, 3, 4, 4}, DataType::UInt32);
        Tensor w0 = g->addTensor({2, 3, 3, 3}, DataType::UInt32);
        auto conv = g->addOp<ConvObj>(i0, w0, nullptr, 1, 1);
        EXPECT_EQ(conv->getOutput()->getDims(), (Shape{1, 2, 4, 4}));
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i0 = g->addTensor({1, 3, 4, 4}, DataType::UInt32);
        Tensor w0 = g->addTensor({2, 3, 3, 3}, DataType::UInt32);
        auto conv =
            g->addOp<ConvObj>(i0, w0, nullptr, ConvObj::PaddingMode::Same);
        EXPECT_EQ(conv->getOutput()->getDims(), (Shape{1, 2, 4, 4}));
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i0 = g->addTensor({1, 3, 4, 4}, DataType::UInt32);
        Tensor w0 = g->addTensor({2, 3, 3, 3}, DataType::UInt32);
        auto conv =
            g->addOp<ConvObj>(i0, w0, nullptr, ConvObj::PaddingMode::Valid);
        EXPECT_EQ(conv->getOutput()->getDims(), (Shape{1, 2, 2, 2}));
    }
    { // dilation & stride
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i0 = g->addTensor({1, 3, 4, 4}, DataType::UInt32);
        Tensor w0 = g->addTensor({2, 3, 3, 3}, DataType::UInt32);
        auto conv = g->addOp<ConvObj>(i0, w0, nullptr, 1, 1, 2, 1, 1, 2);
        EXPECT_EQ(conv->getOutput()->getDims(), (Shape{1, 2, 2, 2}));
    }
}

TEST(Conv, DynamicPlaceholderGeometryIsDeferred) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor input = g->addTensor({1, 3, 1, 1}, DataType::Float32);
    input->setDimDescs(
        {{false, ""}, {false, ""}, {true, "height"}, {true, "width"}});
    Tensor weight = g->addTensor({2, 3, 3, 3}, DataType::Float32);

    auto conv =
        g->addOp<ConvObj>(input, weight, nullptr, ConvObj::PaddingMode::Valid);
    EXPECT_EQ(conv->getOutput()->getDims(), (Shape{1, 2, 1, 1}));

    input->setShape({1, 3, 5, 5});
    g->shape_infer();
    EXPECT_EQ(conv->getOutput()->getDims(), (Shape{1, 2, 3, 3}));
}

TEST(Conv, NaiveCPU) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i0 = g->addTensor({1, 3, 4, 4}, DataType::UInt32);
    Tensor w0 = g->addTensor({2, 3, 3, 3}, DataType::UInt32);
    auto conv = g->addOp<ConvObj>(i0, w0, nullptr, 1, 1, 2, 1, 1, 2);

    g->dataMalloc();
    i0->setData(IncrementalGenerator());
    w0->setData(IncrementalGenerator());
    runtime->run(g, true, true);
    // check answer
    auto ans =
        make_ref<TensorObj>(Shape{1, 2, 2, 2}, DataType::UInt32, runtime);
    ans->dataMalloc();
    ans->copyin(
        vector<uint32_t>{4794, 4386, 8199, 7506, 11274, 10542, 20835, 19656});
    EXPECT_TRUE(conv->getOutput()->equalData(ans));
}

} // namespace infini
