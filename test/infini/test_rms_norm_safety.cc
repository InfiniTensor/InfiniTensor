#include "core/graph.h"
#include "core/infini_runtime.h"
#include "operators/infini_op.h"
#include "operators/rms_norm.h"
#include "test.h"
#include <map>

namespace infini {
TEST(RmsNormSafetyTest, ShortWeightsAreRejectedByBothCpuRuntimePaths) {
    vector<Runtime> runtimes{make_ref<InfiniRuntimeObj>("cpu"),
                             NativeCpuRuntimeObj::getInstance()};
    for (const auto &runtime : runtimes) {
        for (bool naive : {false, true}) {
            auto graph = make_ref<GraphObj>(runtime);
            auto x = graph->addTensor({1, 4}, DataType::Float32);
            auto w = graph->addTensor({4}, DataType::Float32);
            x->setInput();
            w->setInput();
            auto y = graph->addOp<RMSNormObj>(x, w, nullptr)->getOutput();
            y->setOutput();
            graph->dataMalloc(naive);
            x->copyin(vector<float>{1, 2, 3, 4});
            w->copyin(vector<float>(4, 1));
            EXPECT_NO_THROW(runtime->run(graph));
            w->setShape({1});
            // Exercise execution-side validation even if shape_infer is
            // bypassed.
            graph->dataMalloc(naive);
            x->copyin(vector<float>{1, 2, 3, 4});
            w->copyin(vector<float>{1});
            y->copyin(vector<float>(4, 77));
            EXPECT_THROW(runtime->run(graph), std::exception);
            EXPECT_EQ(y->copyout<float>(), vector<float>(4, 77));
        }
    }
}

TEST(RmsNormSafetyTest, GenericNodeRejectsInvalidBoundsShapesAndDtypes) {
    using Bindings = std::map<std::string, TensorVec>;
    const vector<Shape> inputs{{1, 4}, {4}, {1, 2, 2, 4}, {1, 4}, {1, 4}};
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto runtime = make_ref<InfiniRuntimeObj>("cpu");
        auto graph = make_ref<GraphObj>(runtime);
        auto x = graph->addTensor(inputs[i], DataType::Float32);
        auto w = graph->addTensor({i == 0 ? 1 : 4}, i == 4 ? DataType::Double
                                                           : DataType::Float32);
        auto y = graph->addTensor(i == 3 ? Shape{2, 4} : inputs[i],
                                  DataType::Float32);
        x->setInput();
        w->setInput();
        y->setOutput();
        graph->addOp<InfiniOpObj>(
            "RmsNorm", Bindings{{"input", {x}}, {"weight", {w}}},
            Bindings{{"out", {y}}}, nlohmann::json{{"eps", 1e-6}});
        graph->dataMalloc(true);
        y->copyin(vector<float>(y->size(), 77));
        EXPECT_THROW(runtime->run(graph), std::exception);
        EXPECT_EQ(y->copyout<float>(), vector<float>(y->size(), 77));
    }
}
} // namespace infini
