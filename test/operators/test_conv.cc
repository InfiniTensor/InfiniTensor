#include "core/graph.h"
#include "core/run_enigne.h"
#include "operators/conv.h"
#include "test.h"

namespace infini {

TEST(Conv, ShapeInference) {
    // Padding modes
    {
        Graph g = make_ref<GraphObj>();
        Tensor i0 = g->addTensor({1, 3, 4, 4}, DataType::Int32);
        Tensor w0 = g->addTensor({2, 3, 3, 3}, DataType::Int32);
        auto conv = g->addOp<ConvObj>(i0, w0, nullptr, 1, 1);
        EXPECT_EQ(conv->getOutput()->getDims(), (Shape{1, 2, 4, 4}));
    }
    {
        Graph g = make_ref<GraphObj>();
        Tensor i0 = g->addTensor({1, 3, 4, 4}, DataType::Int32);
        Tensor w0 = g->addTensor({2, 3, 3, 3}, DataType::Int32);
        auto conv =
            g->addOp<ConvObj>(i0, w0, nullptr, ConvObj::PaddingMode::Same);
        EXPECT_EQ(conv->getOutput()->getDims(), (Shape{1, 2, 4, 4}));
    }
    {
        Graph g = make_ref<GraphObj>();
        Tensor i0 = g->addTensor({1, 3, 4, 4}, DataType::Int32);
        Tensor w0 = g->addTensor({2, 3, 3, 3}, DataType::Int32);
        auto conv =
            g->addOp<ConvObj>(i0, w0, nullptr, ConvObj::PaddingMode::Valid);
        EXPECT_EQ(conv->getOutput()->getDims(), (Shape{1, 2, 2, 2}));
    }
    { // dilation & stride
        Graph g = make_ref<GraphObj>();
        Tensor i0 = g->addTensor({1, 3, 4, 4}, DataType::Int32);
        Tensor w0 = g->addTensor({2, 3, 3, 3}, DataType::Int32);
        auto conv = g->addOp<ConvObj>(i0, w0, nullptr, 1, 1, 2, 1, 1, 2);
        EXPECT_EQ(conv->getOutput()->getDims(), (Shape{1, 2, 2, 2}));
    }
}

} // namespace infini