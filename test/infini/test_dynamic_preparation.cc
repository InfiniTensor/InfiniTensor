#include "core/graph_handler.h"
#include "core/infini_runtime.h"
#include "operators/gather.h"
#include "operators/layer_norm.h"
#include "test.h"
#include <cmath>
#include <nlohmann/json.hpp>

namespace infini {
#if INFINITENSOR_TEST_HAS_ATEN
namespace {
Ref<InfiniRuntimeObj> makeDynamicRuntime() {
    return make_ref<InfiniRuntimeObj>(
        string(::infini::rt::Device::StringFromType(
            ::infini::rt::runtime_device_type())));
}
TEST(DynamicPreparationTest,
     SameShapeSliceParameterChangeInvalidatesPreparedPlan) {
    auto runtime = makeDynamicRuntime();
    GraphHandlerObj handler(runtime);
    auto x = handler.tensor({8}, 1);
    x->setInput();
    auto y = handler.slice(x, nullptr, {0}, {2}, std::nullopt, std::nullopt);
    y->setOutput();
    handler.data_malloc();
    x->copyin(vector<float>{0, 1, 2, 3, 4, 5, 6, 7});
    handler.run();
    EXPECT_EQ(y->copyout<float>(), (vector<float>{0, 1}));
    auto address = y->getRawDataPtr<void *>();
    handler.update_slice({3}, {5}, std::nullopt, std::nullopt, y->getFuid());
    handler.shape_infer();
    handler.data_malloc();
    EXPECT_EQ(y->getRawDataPtr<void *>(), address);
    handler.run();
    EXPECT_EQ(y->copyout<float>(), (vector<float>{3, 4}));
    EXPECT_THROW(handler.update_slice({0}, {2}, std::nullopt, vector<int>{0},
                                      y->getFuid()),
                 std::exception);
    handler.run();
    EXPECT_EQ(y->copyout<float>(), (vector<float>{3, 4}));
}
TEST(DynamicPreparationTest,
     EncoderAdaptersArePreparedWithoutExecutingOutputs) {
    auto runtime = makeDynamicRuntime();
    auto graph = make_ref<GraphObj>(runtime);
    auto table = graph->addTensor({5, 4}, DataType::Float32);
    auto ids = graph->addTensor({2, 3}, DataType::Int64);
    auto scale = graph->addTensor({4}, DataType::Float32);
    auto bias = graph->addTensor({4}, DataType::Float32);
    table->setWeight();
    scale->setWeight();
    bias->setWeight();
    ids->setInput();
    auto gathered =
        graph->addOp<GatherObj>(table, ids, nullptr, 0)->getOutput();
    auto y =
        graph->addOp<LayerNormObj>(gathered, scale, nullptr, bias, .002f, -1, 1)
            ->getOutput();
    y->setOutput();
    graph->dataMalloc();
    vector<float> weights(20);
    for (int i = 0; i < 20; ++i)
        weights[i] = i;
    table->copyin(weights);
    scale->copyin(vector<float>{.5f, 1, 1.5f, 2});
    bias->copyin(vector<float>{1, 2, 3, 4});
    ids->copyin(vector<int64_t>{0, 1, -1, 3, 2, 0});
    y->copyin(vector<float>(24, 77));
    auto report = nlohmann::json::parse(runtime->checkCapabilities(graph));
    ASSERT_EQ(report["steps"].size(), 2u);
    EXPECT_EQ(report["steps"][0]["operator"], "OnnxGather");
    EXPECT_EQ(report["steps"][1]["operator"], "NativeLayerNorm");
    EXPECT_FALSE(report["steps"][1]["provider_cpp_type"].get<string>().empty());
    EXPECT_EQ(y->copyout<float>(), vector<float>(24, 77));
    runtime->run(graph);
    auto picked = gathered->copyout<float>();
    const vector<int> rows{0, 1, 4, 3, 2, 0};
    for (size_t row = 0; row < rows.size(); ++row)
        for (size_t col = 0; col < 4; ++col)
            EXPECT_EQ(picked[row * 4 + col], weights[rows[row] * 4 + col]);
    auto result = y->copyout<float>();
    for (size_t row = 0; row < 6; ++row)
        for (size_t col = 0; col < 4; ++col)
            EXPECT_NEAR(result[row * 4 + col],
                        (float(col) - 1.5f) / std::sqrt(1.25f + .002f) *
                                (.5f + .5f * col) +
                            (1 + col),
                        1e-5);
}
} // namespace
#endif
} // namespace infini
