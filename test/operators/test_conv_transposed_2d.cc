#include "core/graph.h"
#include "core/kernel.h"
#include "core/perf_engine.h"
#include "core/runtime.h"
#include "operators/conv.h"

#include "test.h"

namespace infini {

TEST(ConvTransposed, ShapeInference) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    { // No pad: InfoGAN ConvTranspose_0
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i0 = g->addTensor({1, 228, 1, 1});
        Tensor w0 = g->addTensor({228, 448, 2, 2});
        auto conv = g->addOp<ConvTransposed2dObj>(i0, w0, nullptr, 0, 0);
        EXPECT_EQ(conv->getOutput()->getDims(), (Shape{1, 448, 2, 2}));
    }
    { // Padded, Strided: InfoGAN ConvTranspose_3
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i0 = g->addTensor({1, 448, 2, 2});
        Tensor w0 = g->addTensor({448, 256, 4, 4});
        auto conv = g->addOp<ConvTransposed2dObj>(i0, w0, nullptr, 1, 1, 2, 2);
        EXPECT_EQ(conv->getOutput()->getDims(), (Shape{1, 256, 4, 4}));
    }
    { // With output padding: GCN ConvTranspose_224
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i0 = g->addTensor({1, 21, 7, 7});
        Tensor w0 = g->addTensor({21, 21, 3, 3});
        auto conv = g->addOp<ConvTransposed2dObj>(i0, w0, nullptr, 1, 1, 2, 2,
                                                  1, 1, 1, 1);
        EXPECT_EQ(conv->getOutput()->getDims(), (Shape{1, 21, 14, 14}));
    }
}

} // namespace infini
