#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace infini {
namespace mlp {
inline void validateDimensions(int batch, int input, int hidden, int output) {
    if (batch < 1 || batch > 1024 || input < 1 || input > 4096 || hidden < 1 ||
        hidden > 4096 || output < 1 || output > 4096)
        throw std::runtime_error("MLP: dimensions outside supported limits");
    size_t elements = size_t(batch) * input + size_t(input) * hidden + hidden +
                      size_t(hidden) * output + output;
    if (elements > 32U * 1024U * 1024U)
        throw std::runtime_error("MLP: request exceeds element limit");
}
inline void validateFinite(const float *data, size_t count) {
    for (size_t i = 0; i < count; ++i)
        if (!std::isfinite(data[i]))
            throw std::runtime_error("MLP: non-finite input or weight");
}
// Pure computation: row-major FP32 Linear -> ReLU -> Linear.
// Accumulate from zero and add bias afterwards, matching separate MatMul/Add
// nodes. Build the target with -ffp-contract=off; no transport or QEMU
// dependency here.
inline void mlp_compute(int batch, int input, int hidden, int output,
                        const float *x, const float *w1, const float *b1,
                        const float *w2, const float *b2, float *logits) {
    validateDimensions(batch, input, hidden, output);
    std::vector<float> activation(size_t(batch) * hidden);
    for (int r = 0; r < batch; ++r)
        for (int j = 0; j < hidden; ++j) {
            float sum = 0;
            for (int k = 0; k < input; ++k)
                sum += x[r * input + k] * w1[k * hidden + j];
            activation[r * hidden + j] = std::max(sum + b1[j], 0.0f);
        }
    for (int r = 0; r < batch; ++r)
        for (int j = 0; j < output; ++j) {
            float sum = 0;
            for (int k = 0; k < hidden; ++k)
                sum += activation[r * hidden + k] * w2[k * output + j];
            logits[r * output + j] = sum + b2[j];
        }
}
} // namespace mlp
} // namespace infini
