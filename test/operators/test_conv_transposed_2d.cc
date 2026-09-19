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

/// A transposed convolution over a dynamic spatial size reads that size afresh
/// every time, so the output grows and shrinks with the input rather than
/// staying at whatever the placeholder gave at construction.
TEST(ConvTransposed, DynamicSpatialDimensionsAreFollowed) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor input = g->addTensor({1, 3, 1, 1}, DataType::Float32);
    input->setDimDescs(
        {{false, ""}, {false, ""}, {true, "height"}, {true, "width"}});
    Tensor weight = g->addTensor({3, 2, 3, 3}, DataType::Float32);

    // Stride two, one of padding either side, one of output padding: the shape
    // ONNX works out for this is 2 * input.
    auto conv = g->addOp<ConvTransposed2dObj>(input, weight, nullptr, 1, 1, 2,
                                              2, 1, 1, 1, 1);
    for (const auto &[given, expected] :
         vector<std::pair<Shape, Shape>>{{{1, 3, 4, 4}, {1, 2, 8, 8}},
                                         {{1, 3, 2, 6}, {1, 2, 4, 12}},
                                         {{1, 3, 7, 5}, {1, 2, 14, 10}}}) {
        input->setShape(given);
        g->shape_infer();
        EXPECT_EQ(conv->getOutput()->getDims(), expected)
            << "input " << vecToString(given);
    }
}

/// Batch follows the batch and the spatial dimensions follow their own place,
/// but the channels come from the weight, which no caller can vary.
TEST(ConvTransposed, ChannelsFollowTheWeightRatherThanTheInput) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor input = g->addTensor({1, 3, 4, 4}, DataType::Float32);
    input->setDimDescs(
        {{true, "batch"}, {false, ""}, {true, "height"}, {true, "width"}});
    Tensor weight = g->addTensor({3, 2, 3, 3}, DataType::Float32);
    // A weight is the data it carries, as an initializer is on the import path.
    weight->dataMalloc();
    weight->setWeight();
    auto conv = g->addOp<ConvTransposed2dObj>(input, weight, nullptr, 0, 0);
    g->shape_infer();

    const auto output = conv->getOutput();
    EXPECT_TRUE(output->isDimDynamic(0));  // batch
    EXPECT_FALSE(output->isDimDynamic(1)); // channels, from the weight
    EXPECT_TRUE(output->isDimDynamic(2));
    EXPECT_TRUE(output->isDimDynamic(3));
}

} // namespace infini
