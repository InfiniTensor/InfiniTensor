#pragma once
#include "core/graph.h"
#include "core/runtime.h"
#include "helpers/platform.h"
#include "test.h"
#include <cmath>
#include <functional>
#include <optional>
#include <vector>

namespace infini {
namespace test {

struct Tolerance {
    double rtol = 1e-5; // relative tolerance
    double atol = 1e-5; // absolute tolerance
};

using OpBuilder = std::function<Operator(GraphObj *, const TensorVec &)>;

inline std::vector<float> runOpAndGetOutput(
    Device::Type deviceType,
    const std::vector<std::pair<Shape, std::vector<float>>> &inputs,
    OpBuilder opBuilder, DataType dtype = DataType::Float32,
    std::optional<std::size_t> implOverride = std::nullopt) {
    Runtime runtime = createRuntime(deviceType);
    if (implOverride.has_value()) {
        runtime->setTestImplOverride(*implOverride);
    }
    Graph g = make_ref<GraphObj>(runtime);

    TensorVec inputTensors;
    inputTensors.reserve(inputs.size());
    for (const auto &[shape, data] : inputs) {
        inputTensors.push_back(g->addTensor(shape, dtype));
    }

    auto op = opBuilder(g.get(), inputTensors);
    g->dataMalloc();

    for (size_t i = 0; i < inputTensors.size(); i++) {
        inputTensors[i]->copyin(inputs[i].second);
    }

    runtime->run(g);
    Tensor output = op->getOutputs()[0];
    return output->copyout<float>();
}

// Cross-platform verification: compute on CPU (reference) and on the target
// platform, then compare with tolerance.
inline void verifyAgainstCpu(
    Device::Type targetType,
    const std::vector<std::pair<Shape, std::vector<float>>> &inputs,
    OpBuilder opBuilder, Tolerance tol = {}) {
    auto cpuResult = runOpAndGetOutput(Device::Type::kCpu, inputs, opBuilder);
    auto targetResult = runOpAndGetOutput(targetType, inputs, opBuilder);

    EXPECT_EQ(cpuResult.size(), targetResult.size())
        << "CPU and target output sizes differ: cpu=" << cpuResult.size()
        << " target=" << targetResult.size();

    for (size_t i = 0; i < cpuResult.size(); i++) {
        double diff = std::fabs(static_cast<double>(cpuResult[i]) -
                                static_cast<double>(targetResult[i]));
        double denom =
            std::max(std::fabs(static_cast<double>(cpuResult[i])),
                     std::fabs(static_cast<double>(targetResult[i])));

        // Use combined tolerance: atol + rtol * |expected|
        // This avoids false failures when values are near zero.
        double threshold = tol.atol + tol.rtol * denom;
        EXPECT_LE(diff, threshold)
            << "Mismatch at index " << i << " / " << cpuResult.size()
            << ": cpu=" << cpuResult[i] << " target=" << targetResult[i]
            << " diff=" << diff << " threshold=" << threshold;
    }
}

// Exact comparison against golden data (for CPU tests).
inline void verifyGoldenData(const std::vector<float> &actual,
                             const std::vector<float> &expected) {
    ASSERT_EQ(actual.size(), expected.size())
        << "Size mismatch: got " << actual.size() << ", expected "
        << expected.size();
    for (size_t i = 0; i < actual.size(); i++) {
        EXPECT_FLOAT_EQ(actual[i], expected[i])
            << "Mismatch at index " << i << " / " << actual.size();
    }
}

} // namespace test
} // namespace infini
