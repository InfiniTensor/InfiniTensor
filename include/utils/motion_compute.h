#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace infini {
namespace motion {
// Contract v1: the delivered Baxter/113-neuron floating-point reservoir.
constexpr int Inputs = 21, Channels = 42, Neurons = 113, Excitatory = 90;
constexpr int Steps = 50, Features = 120, Hidden = 128, Outputs = 7;
inline void require(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
inline void validateSamples(int samples) {
    require(samples > 0 && samples <= 1024,
            "Motion: samples must be in [1,1024]");
}
template <class T> inline void finite(const T *x, size_t n) {
    for (size_t i = 0; i < n; ++i)
        require(std::isfinite(x[i]), "Motion: non-finite tensor");
}
inline void features(int samples, const double *raw, const double *previous,
                     const double *feedback, const double *initial,
                     const double *wi, const double *wr, const double *mean,
                     const double *scale, float *out, double *finalState,
                     double *lastInput, int32_t *counts) {
    validateSamples(samples);
    finite(raw, samples * Inputs);
    finite(previous, Inputs);
    finite(feedback, samples * Outputs);
    finite(initial, 3 * Neurons);
    finite(wi, Channels * Neurons);
    finite(wr, Neurons * Neurons);
    finite(mean, Neurons);
    finite(scale, Neurons);
    for (int n = 0; n < Neurons; ++n)
        require(scale[n] > 0, "Motion: count scale must be positive");
    std::copy(initial, initial + 3 * Neurons, finalState);
    std::copy(previous, previous + Inputs, lastInput);
    std::fill(counts, counts + samples * Neurons, 0);
    double *current = finalState, *voltage = finalState + Neurons;
    double *refractory = finalState + 2 * Neurons;
    for (int sample = 0; sample < samples; ++sample) {
        std::array<int, Channels> remaining{};
        // Feedback spikes and counts reset at sample boundaries; I/V/ref carry.
        std::array<bool, Neurons> previousSpike{};
        for (int j = 0; j < Inputs; ++j) {
            double delta = raw[sample * Inputs + j] - lastInput[j];
            remaining[2 * j + (delta < 0)] =
                int(std::min(50.0, std::ceil(std::abs(delta) / 1e-5)));
            lastInput[j] = raw[sample * Inputs + j];
        }
        for (int step = 0; step < Steps; ++step) {
            std::array<double, Neurons> inputCurrent{}, recurrentCurrent{};
            for (int j = 0; j < Channels; ++j) {
                if (remaining[j] <= 0)
                    continue;
                --remaining[j];
                double sign = j % 2 ? -1.0 : 1.0;
                for (int n = 0; n < Neurons; ++n)
                    inputCurrent[n] += sign * wi[j * Neurons + n];
            }
            for (int j = 0; j < Neurons; ++j) {
                if (!previousSpike[j])
                    continue;
                for (int n = 0; n < Neurons; ++n)
                    recurrentCurrent[n] += wr[j * Neurons + n];
            }
            for (int n = 0; n < Neurons; ++n) {
                current[n] += (inputCurrent[n] + recurrentCurrent[n]) * 1024.0;
                if (refractory[n] > 0) {
                    voltage[n] -= std::floor(voltage[n] / 128.0);
                    refractory[n] -= 0.1;
                } else {
                    voltage[n] = voltage[n] - std::floor(voltage[n] / 128.0) +
                                 std::floor(current[n]);
                    current[n] -= std::floor(current[n] / 16.0);
                }
                previousSpike[n] =
                    voltage[n] >= (n < Excitatory ? 10240.0 : 15360.0);
                if (previousSpike[n]) {
                    voltage[n] = current[n] = 0;
                    refractory[n] = n < Excitatory ? 0.3 : 0.5;
                    ++counts[sample * Neurons + n];
                }
            }
        }
        for (int n = 0; n < Neurons; ++n)
            out[sample * Features + n] =
                float((counts[sample * Neurons + n] - mean[n]) / scale[n]);
        for (int j = 0; j < Outputs; ++j)
            out[sample * Features + Neurons + j] =
                float(feedback[sample * Outputs + j]);
    }
    finite(finalState, 3 * Neurons);
    finite(out, samples * Features);
}
inline void readout(int samples, const float *x, const float *w1,
                    const float *b1, const float *w2, const float *b2,
                    const double *mean, const double *scale, double *out) {
    validateSamples(samples);
    finite(x, samples * Features);
    finite(w1, Features * Hidden);
    finite(b1, Hidden);
    finite(w2, Hidden * Outputs);
    finite(b2, Outputs);
    finite(mean, Outputs);
    finite(scale, Outputs);
    for (int j = 0; j < Outputs; ++j)
        require(scale[j] > 0, "Motion: output scale must be positive");
    for (int sample = 0; sample < samples; ++sample) {
        std::array<float, Hidden> activation;
        for (int j = 0; j < Hidden; ++j) {
            float sum = 0;
            for (int k = 0; k < Features; ++k)
                sum += x[sample * Features + k] * w1[k * Hidden + j];
            activation[j] = std::tanh(sum + b1[j]);
        }
        for (int j = 0; j < Outputs; ++j) {
            float sum = 0;
            for (int k = 0; k < Hidden; ++k)
                sum += activation[k] * w2[k * Outputs + j];
            out[sample * Outputs + j] =
                double(sum + b2[j]) * scale[j] + mean[j];
        }
    }
    finite(out, samples * Outputs);
}
} // namespace motion
} // namespace infini
