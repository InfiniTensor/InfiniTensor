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
    auto graph = make_ref<GraphObj>(NativeCpuRuntimeObj::getInstance());
    auto x = graph->addTensor({1, 4}, DataType::Float32);
    auto wrongType = graph->addTensor({4}, DataType::Double);
    EXPECT_THROW(graph->addOp<RMSNormObj>(x, wrongType, nullptr),
                 std::exception);
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
