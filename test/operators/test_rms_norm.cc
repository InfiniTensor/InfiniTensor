#include "core/graph.h"
#include "operators/rms_norm.h"
#include "test.h"

namespace infini {
TEST(RmsNormShapeTest, RejectsInvalidInputsDuringConstruction) {
    for (const auto &shape : vector<Shape>{{}, {1, 0}, {1, 4}}) {
        auto graph = make_ref<GraphObj>(NativeCpuRuntimeObj::getInstance());
        auto x = graph->addTensor(shape, DataType::Float32);
        auto shortWeight = graph->addTensor({1}, DataType::Float32);
        EXPECT_THROW(graph->addOp<RMSNormObj>(x, shortWeight, nullptr),
                     std::exception);
    }
}

TEST(RmsNormShapeTest, MixedPrecisionDoesNotRestrictShapeInference) {
    for (auto dtype : {DataType::Float16, DataType::BFloat16}) {
        auto graph = make_ref<GraphObj>(NativeCpuRuntimeObj::getInstance());
        auto x = graph->addTensor({2, 128}, dtype);
        auto w = graph->addTensor({128}, DataType::Float32);
        auto y = graph->addOp<RMSNormObj>(x, w, nullptr)->getOutput();
        EXPECT_EQ(y->getDims(), x->getDims());
        EXPECT_EQ(y->getDType(), dtype);
        EXPECT_NO_THROW(graph->shape_infer());
    }
}

TEST(RmsNormShapeTest, RevalidatesWeightsAfterDynamicShapeChanges) {
    auto graph = make_ref<GraphObj>(NativeCpuRuntimeObj::getInstance());
    auto x = graph->addTensor({1, 4}, DataType::Float32);
    auto w = graph->addTensor({4}, DataType::Float32);
    graph->addOp<RMSNormObj>(x, w, nullptr);
    w->setShape({1});
    EXPECT_THROW(graph->shape_infer(), std::exception);
    w->setShape({4});
    EXPECT_NO_THROW(graph->shape_infer());
}
} // namespace infini
