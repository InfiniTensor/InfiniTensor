#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/conv.h"

#include "test.h"

namespace infini {

TEST(Conv3d, ShapeInference) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    // Padding modes
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i0 = g->addTensor({1, 3, 4, 4, 4}, DataType::UInt32);
        Tensor w0 = g->addTensor({2, 3, 3, 3, 3}, DataType::UInt32);
        auto conv3d = g->addOp<Conv3dObj>(i0, w0, nullptr, 1, 1, 1);
        EXPECT_EQ(conv3d->getOutput()->getDims(), (Shape{1, 2, 4, 4, 4}));
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i0 = g->addTensor({1, 3, 4, 4, 4}, DataType::UInt32);
        Tensor w0 = g->addTensor({2, 3, 3, 3, 3}, DataType::UInt32);
        auto conv3d =
            g->addOp<Conv3dObj>(i0, w0, nullptr, Conv3dObj::PaddingMode::Same);
        EXPECT_EQ(conv3d->getOutput()->getDims(), (Shape{1, 2, 4, 4, 4}));
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i0 = g->addTensor({1, 3, 4, 4, 4}, DataType::UInt32);
        Tensor w0 = g->addTensor({2, 3, 3, 3, 3}, DataType::UInt32);
        auto conv3d =
            g->addOp<Conv3dObj>(i0, w0, nullptr, Conv3dObj::PaddingMode::Valid);
        EXPECT_EQ(conv3d->getOutput()->getDims(), (Shape{1, 2, 2, 2, 2}));
    }
    { // dilation & stride
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i0 = g->addTensor({1, 3, 4, 4, 4}, DataType::UInt32);
        Tensor w0 = g->addTensor({2, 3, 3, 3, 3}, DataType::UInt32);
        auto conv3d =
            g->addOp<Conv3dObj>(i0, w0, nullptr, 1, 1, 1, 1, 2, 1, 1, 1, 2);
        EXPECT_EQ(conv3d->getOutput()->getDims(), (Shape{1, 2, 4, 2, 2}));
    }
}

} // namespace infini
