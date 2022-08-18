#include "core/graph.h"
#include "core/run_enigne.h"
#include "operators/conv.h"
#include "test.h"

namespace infini {

TEST(Conv, ShapeInference) {
    // Padding modes
    {
        Graph g = make_ref<GraphObj>();
        Tensor i0 = g->addTensor({1, 3, 4, 4}, DataType::UInt32);
        Tensor w0 = g->addTensor({2, 3, 3, 3}, DataType::UInt32);
        auto conv = g->addOp<ConvObj>(i0, w0, nullptr, 1, 1);
        EXPECT_EQ(conv->getOutput()->getDims(), (Shape{1, 2, 4, 4}));
    }
    {
        Graph g = make_ref<GraphObj>();
        Tensor i0 = g->addTensor({1, 3, 4, 4}, DataType::UInt32);
        Tensor w0 = g->addTensor({2, 3, 3, 3}, DataType::UInt32);
        auto conv =
            g->addOp<ConvObj>(i0, w0, nullptr, ConvObj::PaddingMode::Same);
        EXPECT_EQ(conv->getOutput()->getDims(), (Shape{1, 2, 4, 4}));
    }
    {
        Graph g = make_ref<GraphObj>();
        Tensor i0 = g->addTensor({1, 3, 4, 4}, DataType::UInt32);
        Tensor w0 = g->addTensor({2, 3, 3, 3}, DataType::UInt32);
        auto conv =
            g->addOp<ConvObj>(i0, w0, nullptr, ConvObj::PaddingMode::Valid);
        EXPECT_EQ(conv->getOutput()->getDims(), (Shape{1, 2, 2, 2}));
    }
    { // dilation & stride
        Graph g = make_ref<GraphObj>();
        Tensor i0 = g->addTensor({1, 3, 4, 4}, DataType::UInt32);
        Tensor w0 = g->addTensor({2, 3, 3, 3}, DataType::UInt32);
        auto conv = g->addOp<ConvObj>(i0, w0, nullptr, 1, 1, 2, 1, 1, 2);
        EXPECT_EQ(conv->getOutput()->getDims(), (Shape{1, 2, 2, 2}));
    }
}

TEST(Conv, NaiveCPU) {
    Graph g = make_ref<GraphObj>();
    Tensor i0 = g->addTensor({1, 3, 4, 4}, DataType::UInt32);
    Tensor w0 = g->addTensor({2, 3, 3, 3}, DataType::UInt32);
    auto conv = g->addOp<ConvObj>(i0, w0, nullptr, 1, 1, 2, 1, 1, 2);

    g->dataMalloc();
    i0->setData(IncrementalGenerator());
    w0->setData(IncrementalGenerator());
    RunEngine(Device::CPU).run(g, true, true);
    double perfTime = RunEngine(Device::CPU).getPerfTime(g);
    // The example Conv takes 0.015ms with one core
    EXPECT_GT(perfTime, 0);
    EXPECT_LT(perfTime, 0.1);
    // check answer
    auto ans = make_ref<TensorObj>(Shape{1, 2, 2, 2}, DataType::UInt32);
    ans->dataMalloc();
    ans->copyData({4794, 4386, 8199, 7506, 11274, 10542, 20835, 19656});
    EXPECT_TRUE(conv->getOutput()->equalData(ans));
}

} // namespace infini