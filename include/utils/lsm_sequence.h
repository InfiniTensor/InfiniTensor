#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace infini {
namespace lsm {
constexpr int StateRows = 5, ParameterCount = 21;
enum StateRow { Current, Voltage, Refractory, PreviousSpike, Count };
enum Param {
    ExcitatoryCount,
    WeightShift,
    ThresholdI,
    ThresholdE,
    PeriodI,
    PeriodE,
    VoltageShiftI,
    VoltageShiftE,
    CurrentShiftI,
    CurrentShiftE,
    VIShiftI,
    VIShiftE,
    RoundVoltageI,
    RoundVoltageE,
    RoundCurrentI,
    RoundCurrentE,
    ResetVoltageI,
    ResetVoltageE,
    ResetCurrentI,
    ResetCurrentE,
    CountMode
};
using Parameters = std::array<int32_t, ParameterCount>;
inline void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
inline void validateParameters(const Parameters &p, int neurons) {
    require(p[ExcitatoryCount] >= 0 && p[ExcitatoryCount] <= neurons,
            "LSMSequence: invalid excitatory_count");
    require(p[WeightShift] >= 0 && p[WeightShift] <= 16,
            "LSMSequence: weight_shift must be in [0,16]");
    for (int k = VoltageShiftI; k <= RoundCurrentE; ++k)
        require(p[k] >= 0 && p[k] <= 31,
                "LSMSequence: shifts must be in [0,31]");
    require(p[PeriodI] >= 0 && p[PeriodI] <= 255 && p[PeriodE] >= 0 &&
                p[PeriodE] <= 255,
            "LSMSequence: refractory periods must be in [0,255]");
    require(p[CountMode] == 0 || p[CountMode] == 1,
            "LSMSequence: count_mode must be 0 (wide) or 1 (uint8 wrap)");
}
inline void validateInputs(int steps, int channels, int neurons,
                           const Parameters &p, const int32_t *spikes,
                           const int32_t *inputWeights,
                           const int32_t *recurrent, const int32_t *state) {
    require(steps > 0 && steps <= 6000 && channels > 0 && channels <= 1024 &&
                neurons > 0 && neurons <= 1024,
            "LSMSequence: unsupported dimensions");
    validateParameters(p, neurons);
    for (size_t j = 0; j < size_t(steps) * channels; ++j)
        require(spikes[j] == 0 || spikes[j] == 1,
                "LSMSequence: input spikes must be binary");
    for (size_t j = 0; j < size_t(channels) * neurons; ++j)
        require(inputWeights[j] >= -8 && inputWeights[j] <= 7,
                "LSMSequence: input weights must be signed INT4");
    for (size_t j = 0; j < size_t(neurons) * neurons; ++j)
        require(recurrent[j] >= -8 && recurrent[j] <= 7,
                "LSMSequence: recurrent weights must be signed INT4");
    for (int j = 0; j < neurons; ++j) {
        require(state[Refractory * neurons + j] >= 0 &&
                    state[Refractory * neurons + j] <= 255,
                "LSMSequence: invalid NPR state");
        require(state[PreviousSpike * neurons + j] == 0 ||
                    state[PreviousSpike * neurons + j] == 1,
                "LSMSequence: invalid previous spike");
        int32_t count = state[Count * neurons + j];
        require(
            count >= 0 &&
                (p[CountMode] == 1
                     ? count <= 255
                     : count <= std::numeric_limits<int32_t>::max() - steps),
            "LSMSequence: invalid count state or count overflow");
    }
}
inline int32_t signedWord(uint32_t word) {
    return word <= 0x7fffffffU ? int32_t(word)
                               : int32_t(int64_t(word) - 0x100000000LL);
}
inline int32_t wrap(int64_t value) { return signedWord(uint32_t(value)); }
inline int32_t ashr(int32_t value, int shift) {
    if (!shift)
        return value;
    uint32_t word = uint32_t(value) >> shift;
    if (value < 0)
        word |= 0xffffffffU << (32 - shift);
    return signedWord(word);
}
inline void featuresFromState(const int32_t *state, int neurons,
                              const Parameters &p, float *features) {
    const auto *counts = state + Count * neurons;
    for (int group = 0; group < 2; ++group) {
        int start = group == 0 ? 0 : p[ExcitatoryCount];
        int end = group == 0 ? p[ExcitatoryCount] : neurons;
        int32_t maximum = 0;
        for (int j = start; j < end; ++j)
            maximum = std::max(maximum, counts[j]);
        for (int j = start; j < end; ++j)
            features[j] = maximum ? float(counts[j]) / float(maximum) : 0.f;
    }
}
inline void runReference(int steps, int channels, int neurons,
                         const Parameters &p, const int32_t *spikes,
                         const int32_t *inputWeights, const int32_t *recurrent,
                         const int32_t *initial, bool traceAll, float *features,
                         int32_t *finalState, int32_t *trace) {
    validateInputs(steps, channels, neurons, p, spikes, inputWeights, recurrent,
                   initial);
    std::vector<int32_t> state(initial, initial + StateRows * neurons);
    std::vector<int32_t> next(StateRows * neurons);
    for (int t = 0; t < steps; ++t) {
        for (int j = 0; j < neurons; ++j) {
            int64_t sum = 0;
            for (int k = 0; k < channels; ++k)
                if (spikes[t * channels + k])
                    sum += inputWeights[k * neurons + j];
            for (int k = 0; k < neurons; ++k)
                if (state[PreviousSpike * neurons + k])
                    sum += recurrent[k * neurons + j];
            int32_t i = wrap(int64_t(state[Current * neurons + j]) +
                             sum * (1LL << p[WeightShift]));
            int32_t v = state[Voltage * neurons + j];
            int32_t r = state[Refractory * neurons + j];
            int32_t count = state[Count * neurons + j];
            bool excited = j < p[ExcitatoryCount];
            int32_t ni =
                wrap(int64_t(i) -
                     ashr(ashr(i, p[excited ? CurrentShiftE : CurrentShiftI]),
                          p[excited ? RoundCurrentE : RoundCurrentI]));
            int32_t nv = v, fired = 0;
            if (r) {
                --r;
            } else {
                nv = wrap(
                    int64_t(v) -
                    ashr(ashr(v, p[excited ? VoltageShiftE : VoltageShiftI]),
                         p[excited ? RoundVoltageE : RoundVoltageI]) +
                    ashr(i, p[excited ? VIShiftE : VIShiftI]));
                if (nv > p[excited ? ThresholdE : ThresholdI]) {
                    fired = 1;
                    nv = p[excited ? ResetVoltageE : ResetVoltageI];
                    ni = p[excited ? ResetCurrentE : ResetCurrentI];
                    r = p[excited ? PeriodE : PeriodI];
                    count = p[CountMode] == 1 ? (count + 1) & 255 : count + 1;
                }
            }
            next[Current * neurons + j] = ni;
            next[Voltage * neurons + j] = nv;
            next[Refractory * neurons + j] = r;
            next[PreviousSpike * neurons + j] = fired;
            next[Count * neurons + j] = count;
        }
        state.swap(next);
        if (traceAll)
            std::copy(state.begin(), state.end(),
                      trace + size_t(t) * StateRows * neurons);
    }
    std::copy(state.begin(), state.end(), finalState);
    if (!traceAll)
        std::copy(state.begin(), state.end(), trace);
    featuresFromState(finalState, neurons, p, features);
}
} // namespace lsm
} // namespace infini
