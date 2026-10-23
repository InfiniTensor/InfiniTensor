#include "rvne_builtin.h"
#include "utils/lsm_sequence.h"
namespace infini {
namespace lsm {
static std::vector<uint32_t> packWeights(const int32_t *weights, int fanIn,
                                         int neurons) {
    int words = ((fanIn + 127) / 128) * 16;
    std::vector<uint32_t> packed(size_t(neurons) * words, 0);
    for (int n = 0; n < neurons; ++n)
        for (int j = 0; j < fanIn; ++j)
            packed[n * words + j / 8] |= uint32_t(weights[j * neurons + n] & 15)
                                         << (4 * (j % 8));
    return packed;
}
static void currents(const int32_t *spikes, int fanIn, int neurons,
                     const std::vector<uint32_t> &weights) {
    uint32_t packed[32] = {};
    bool active = false;
    for (int j = 0; j < fanIn; ++j) {
        packed[j / 32] |= uint32_t(spikes[j]) << (j % 32);
        active |= spikes[j] != 0;
    }
    if (!active)
        return;
    for (int j = 0; j < 32; ++j)
        rvne_write_svr(packed[j], j);
    int blocks = (fanIn + 127) / 128, words = blocks * 16;
    for (int n = 0; n < neurons; ++n) {
        for (int j = 0; j < words; ++j)
            rvne_write_wvr(weights[n * words + j], j);
        for (int j = 0; j < blocks; ++j)
            rvne_current_acc_128(j, j, n);
    }
}
inline void lsm_compute(int t, int d, int n, const Parameters &p,
                        const int32_t *spikes, const int32_t *wi,
                        const int32_t *wr, const int32_t *initial,
                        bool traceAll, float *features, int32_t *finalState,
                        int32_t *trace) {
    auto inputPacked = packWeights(wi, d, n),
         recurrentPacked = packWeights(wr, n, n);
    std::vector<int32_t> state(initial, initial + StateRows * n);
    rvne_clear_neuron_data_1024();
    for (int group = 0; group < 32; ++group) {
        uint32_t types = 0;
        for (int bit = 0; bit < 32; ++bit)
            if (group * 32 + bit < p[ExcitatoryCount])
                types |= 1U << bit;
        rvne_set_neuron_type_32(group, types);
    }
    rvne_set_voltage_threshold(p[ThresholdI], p[ThresholdE]);
    rvne_set_refractory_period(p[PeriodI], p[PeriodE]);
    rvne_set_reset_voltage(p[ResetVoltageI], p[ResetVoltageE]);
    rvne_set_reset_current(p[ResetCurrentI], p[ResetCurrentE]);
    rvne_set_leakage_index(p[WeightShift], p[VoltageShiftI], p[CurrentShiftI],
                           p[RoundVoltageI], p[RoundCurrentI], p[VIShiftI],
                           p[WeightShift], p[VoltageShiftE], p[CurrentShiftE],
                           p[RoundVoltageE], p[RoundCurrentE], p[VIShiftE]);
    for (int j = 0; j < n; ++j) {
        rvne_write_ncr(state[Current * n + j], j);
        rvne_write_nvr(state[Voltage * n + j], j);
    }
    for (int group = 0; group < (n + 7) / 8; ++group) {
        uint64_t word = 0;
        for (int bit = 0; bit < 8 && group * 8 + bit < n; ++bit)
            word |= uint64_t(state[Refractory * n + group * 8 + bit])
                    << (8 * bit);
        rvne_write_npr_group8(word, group);
    }
    for (int step = 0; step < t; ++step) {
        currents(spikes + step * d, d, n, inputPacked);
        currents(state.data() + PreviousSpike * n, n, n, recurrentPacked);
        // NPR does not guarantee clearing SOR while a neuron is refractory.
        for (int group = 0; group < 16; ++group)
            rvne_write_sor_group2(0, group);
        rvne_clear_neuron_state_1024();
        for (int group = 0; group < (n + 31) / 32; ++group)
            rvne_leakage_integral_fire_32(group, group);
        for (int j = 0; j < n; j += 2) {
            uint64_t i = rvne_read_ncr_group2(j / 2),
                     v = rvne_read_nvr_group2(j / 2);
            for (int k = 0; k < 2 && j + k < n; ++k) {
                state[Current * n + j + k] =
                    signedWord(uint32_t(i >> (32 * k)));
                state[Voltage * n + j + k] =
                    signedWord(uint32_t(v >> (32 * k)));
            }
        }
        for (int j = 0; j < n; j += 8) {
            uint64_t ref = rvne_read_npr_group8(j / 8),
                     count = rvne_read_nsr_group8(j / 8);
            for (int k = 0; k < 8 && j + k < n; ++k) {
                state[Refractory * n + j + k] = (ref >> (8 * k)) & 255;
                int increment = (count >> (8 * k)) & 255;
                int32_t &total = state[Count * n + j + k];
                total = p[CountMode] ? (total + increment) & 255
                                     : total + increment;
            }
        }
        for (int j = 0; j < n; j += 64) {
            uint64_t output = rvne_read_sor_group2(j / 64);
            for (int k = 0; k < 64 && j + k < n; ++k)
                state[PreviousSpike * n + j + k] = (output >> k) & 1;
        }
        if (traceAll)
            std::copy(state.begin(), state.end(),
                      trace + size_t(step) * StateRows * n);
    }
    std::copy(state.begin(), state.end(), finalState);
    if (!traceAll)
        std::copy(state.begin(), state.end(), trace);
    featuresFromState(finalState, n, p, features);
}

} // namespace lsm
} // namespace infini

#include "utils/mlp_compute.h"
#include "utils/rvne_session.h"
#include <cstring>
#include <iostream>
#include <sstream>

using infini::rvne::read;
using infini::rvne::require;
using infini::rvne::write;

static void lsmRequest(std::istream &input, std::ostream &output) {
    using namespace infini::lsm;
    uint32_t dims[4];
    read(input, dims, 4);
    int t = int(dims[0]), d = int(dims[1]), n = int(dims[2]);
    infini::rvne::require(t > 0 && t <= 6000 && d > 0 && d <= 1024 && n > 0 &&
                              n <= 1024 && dims[3] <= 1,
                          "runner: unsupported dimensions");
    bool traceAll = dims[3];
    Parameters parameters;
    read(input, parameters.data(), parameters.size());
    std::vector<int32_t> spikes(size_t(t) * d), wi(size_t(d) * n),
        wr(size_t(n) * n), initial(StateRows * n);
    read(input, spikes.data(), spikes.size());
    read(input, wi.data(), wi.size());
    read(input, wr.data(), wr.size());
    read(input, initial.data(), initial.size());
    infini::rvne::end(input);
    validateInputs(t, d, n, parameters, spikes.data(), wi.data(), wr.data(),
                   initial.data());
    std::vector<float> features(n);
    std::vector<int32_t> finalState(StateRows * n),
        trace(size_t(traceAll ? t : 1) * StateRows * n);
    lsm_compute(t, d, n, parameters, spikes.data(), wi.data(), wr.data(),
                initial.data(), traceAll, features.data(), finalState.data(),
                trace.data());
    write(output, features.data(), features.size());
    write(output, finalState.data(), finalState.size());
    write(output, trace.data(), trace.size());
}
static void mlpRequest(std::istream &input, std::ostream &output) {
    uint32_t dims[4];
    read(input, dims, 4);
    int batch = int(dims[0]), in = int(dims[1]), hidden = int(dims[2]),
        out = int(dims[3]);
    infini::mlp::validateDimensions(batch, in, hidden, out);
    std::vector<float> x(size_t(batch) * in), w1(size_t(in) * hidden),
        b1(hidden), w2(size_t(hidden) * out), b2(out),
        logits(size_t(batch) * out);
    for (auto *v : {&x, &w1, &b1, &w2, &b2}) {
        read(input, v->data(), v->size());
        infini::mlp::validateFinite(v->data(), v->size());
    }
    infini::rvne::end(input);
    infini::mlp::mlp_compute(batch, in, hidden, out, x.data(), w1.data(),
                             b1.data(), w2.data(), b2.data(), logits.data());
    write(output, logits.data(), logits.size());
}
static void serve() {
    using namespace infini::rvne;
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    uint32_t lastId = 0;
    while (std::cin.peek() != std::char_traits<char>::eof()) {
        Header request;
        read(std::cin, request.data(), request.size());
        infini::rvne::require(request[0] == RequestMagic &&
                                  request[1] == Version &&
                                  request[4] <= MaxPayload && request[5] == 0 &&
                                  request[3] == lastId + 1,
                              "runner: invalid frame header");
        lastId = request[3];
        std::string payload(request[4], '\0');
        read(std::cin, payload.data(), payload.size());
        std::istringstream input(payload, std::ios::binary);
        std::ostringstream output(std::ios::binary);
        uint32_t status = 0;
        try {
            if (request[2] == LSM)
                lsmRequest(input, output);
            else if (request[2] == MLP)
                mlpRequest(input, output);
            else
                throw std::runtime_error("runner: unsupported command");
            payload = output.str();
            infini::rvne::require(payload.size() <= MaxPayload,
                                  "runner: response exceeds limit");
        } catch (const std::exception &error) {
            status = 1;
            payload = std::string(error.what()).substr(0, 8192);
        }
        Header response{ResponseMagic,
                        Version,
                        request[2],
                        request[3],
                        uint32_t(payload.size()),
                        status};
        write(std::cout, response.data(), response.size());
        write(std::cout, payload.data(), payload.size());
        std::cout.flush();
        infini::rvne::require(bool(std::cout), "runner: response flush failed");
    }
    infini::rvne::require(!std::cin.bad(), "runner: channel read failed");
}
int main(int argc, char **argv) {
    try {
        infini::rvne::require(argc == 2 && std::strcmp(argv[1], "--serve") == 0,
                              "usage: lsm_runner --serve");
        serve();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
