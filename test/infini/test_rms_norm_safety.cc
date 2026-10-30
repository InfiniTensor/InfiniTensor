#include "core/graph.h"
#include "core/infini_runtime.h"
#include "operators/infini_op.h"
#include "operators/rms_norm.h"
#include "test.h"
#include "utils/data_convert.h"
#include <cmath>
#include <cstring>
#include <map>

namespace infini {
namespace {
// Preserve the selected SDK backend before CPU-only cases activate CPU.
const auto selectedDevice = ::infini::rt::runtime_device_type();
} // namespace

TEST(RmsNormSafetyTest, MixedPrecisionUsesBackendCapabilities) {
    const auto device = selectedDevice;
    const string name(::infini::rt::Device::StringFromType(device));
    auto runtime = make_ref<InfiniRuntimeObj>(name);
    for (auto dtype : {DataType::Float16, DataType::BFloat16}) {
        const auto encode =
            dtype == DataType::Float16 ? float_to_fp16 : float_to_bfp16;
        const auto decode = [dtype](uint16_t bits) {
            if (dtype == DataType::Float16)
                return fp16_to_float(bits);
            const uint32_t value = uint32_t(bits) << 16;
            float result;
            std::memcpy(&result, &value, sizeof(result));
            return result;
        };
        for (const auto &shape : vector<Shape>{{2, 128}, {2, 3, 128}}) {
            auto graph = make_ref<GraphObj>(runtime);
            auto x = graph->addTensor(shape, dtype);
            auto w = graph->addTensor({128}, DataType::Float32);
            x->setInput();
            w->setWeight();
            auto y = graph->addOp<RMSNormObj>(x, w, nullptr)->getOutput();
            y->setOutput();
            graph->dataMalloc();
            vector<uint16_t> input(x->size()), sentinel(x->size(), encode(77));
            vector<float> weight(128);
            for (size_t i = 0; i < input.size(); ++i)
                input[i] = encode((static_cast<int>(i % 17) - 8) / 8.f);
            for (size_t i = 0; i < weight.size(); ++i)
                weight[i] = 0.7f + (i % 11) * 0.0753f;
            x->copyin(input);
            w->copyin(weight);
            y->copyin(sentinel);
            if (device == ::infini::rt::Device::Type::kCpu) {
                EXPECT_THROW(runtime->run(graph), std::exception);
                EXPECT_EQ(y->copyout<uint16_t>(), sentinel);
                continue;
            }
            ASSERT_NO_THROW(runtime->run(graph));
            const auto actual = y->copyout<uint16_t>();
            for (size_t row = 0; row < input.size() / 128; ++row) {
                float squared = 0;
                for (size_t col = 0; col < 128; ++col) {
                    const float value = decode(input[row * 128 + col]);
                    squared += value * value;
                }
                const float inverse = 1.f / std::sqrt(squared / 128 + 1e-6f);
                for (size_t col = 0; col < 128; ++col) {
                    const auto i = row * 128 + col;
                    const float expected =
                        decode(input[i]) * weight[col] * inverse;
                    EXPECT_NEAR(decode(actual[i]), expected,
                                dtype == DataType::Float16 ? 0.003f : 0.03f);
                }
            }
            std::cout << nlohmann::json{{"device", name},
                                        {"shape", shape},
                                        {"input", dtype == DataType::Float16
                                                      ? "float16"
                                                      : "bfloat16"},
                                        {"weight", "float32"}}
                             .dump()
                      << std::endl;
        }
    }
}

TEST(RmsNormSafetyTest, BothCpuRuntimePathsRejectUnsupportedMixedPrecision) {
    for (const auto &runtime :
         vector<Runtime>{make_ref<InfiniRuntimeObj>("cpu"),
                         NativeCpuRuntimeObj::getInstance()}) {
        for (auto dtype : {DataType::Float16, DataType::BFloat16}) {
            auto graph = make_ref<GraphObj>(runtime);
            auto x = graph->addTensor({1, 128}, dtype);
            auto w = graph->addTensor({128}, DataType::Float32);
            x->setInput();
            w->setInput();
            auto y = graph->addOp<RMSNormObj>(x, w, nullptr)->getOutput();
            y->setOutput();
            graph->dataMalloc(true);
            const auto encode =
                dtype == DataType::Float16 ? float_to_fp16 : float_to_bfp16;
            x->copyin(vector<uint16_t>(128, encode(1)));
            w->copyin(vector<float>(128, 1));
            const vector<uint16_t> sentinel(128, encode(77));
            y->copyin(sentinel);
            EXPECT_THROW(runtime->run(graph), std::exception);
            EXPECT_EQ(y->copyout<uint16_t>(), sentinel);
        }
    }
}

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
