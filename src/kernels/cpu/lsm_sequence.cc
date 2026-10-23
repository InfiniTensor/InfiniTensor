#include "operators/lsm_sequence.h"
#include "core/kernel.h"
#include "utils/mlp_compute.h"
#include "utils/rvne_session.h"
#include <chrono>
#include <cstring>
#include <sstream>

namespace infini {
// The host validates tensors and submits either operation to the same session.
// Only the explicit reference branches below compute on the host.
class RVNESequenceCpu : public CpuKernelWithoutConfig {
    void compute(const Operator &generic, const RuntimeObj *) const override {
        rvne::Command command;
        std::ostringstream request(std::ios::binary);
        if (generic->getOpType() == OpType::LSMSequence) {
            auto op = as<LSMSequenceObj>(generic);
            int t = op->getInputs(0)->getDims()[0];
            int d = op->getInputs(0)->getDims()[1];
            int n = op->getInputs(2)->getDims()[0];
            auto *spikes = op->getInputs(0)->getRawDataPtr<int32_t *>();
            auto *wi = op->getInputs(1)->getRawDataPtr<int32_t *>();
            auto *wr = op->getInputs(2)->getRawDataPtr<int32_t *>();
            auto *state = op->getInputs(3)->getRawDataPtr<int32_t *>();
            const auto &params = op->getParameters();
            lsm::validateInputs(t, d, n, params, spikes, wi, wr, state);
            if (!op->executesQemu()) {
                lsm::runReference(t, d, n, params, spikes, wi, wr, state,
                                  op->tracesAll(),
                                  op->getOutput(0)->getRawDataPtr<float *>(),
                                  op->getOutput(1)->getRawDataPtr<int32_t *>(),
                                  op->getOutput(2)->getRawDataPtr<int32_t *>());
                return;
            }
            command = rvne::LSM;
            uint32_t dims[4] = {uint32_t(t), uint32_t(d), uint32_t(n),
                                uint32_t(op->tracesAll())};
            rvne::write(request, dims, 4);
            rvne::write(request, params.data(), params.size());
        } else {
            auto op = as<MLPObj>(generic);
            const auto x = op->getInputs(0)->getDims();
            uint32_t dims[4] = {uint32_t(x[0]), uint32_t(x[1]),
                                uint32_t(op->getInputs(1)->getDims()[1]),
                                uint32_t(op->getInputs(3)->getDims()[1])};
            mlp::validateDimensions(dims[0], dims[1], dims[2], dims[3]);
            for (const auto &input : op->getInputs())
                mlp::validateFinite(input->getRawDataPtr<float *>(),
                                    input->size());
            if (!op->executesQemu()) {
                mlp::mlp_compute(dims[0], dims[1], dims[2], dims[3],
                                 op->getInputs(0)->getRawDataPtr<float *>(),
                                 op->getInputs(1)->getRawDataPtr<float *>(),
                                 op->getInputs(2)->getRawDataPtr<float *>(),
                                 op->getInputs(3)->getRawDataPtr<float *>(),
                                 op->getInputs(4)->getRawDataPtr<float *>(),
                                 op->getOutput()->getRawDataPtr<float *>());
                return;
            }
            command = rvne::MLP;
            rvne::write(request, dims, 4);
        }
        for (const auto &input : generic->getInputs())
            rvne::write(request, input->getRawDataPtr<char *>(),
                        input->getBytes());
        size_t bytes = 0;
        for (const auto &output : generic->getOutputs())
            bytes += output->getBytes();
        rvne::SessionScope fallback;
        auto response =
            rvne::SessionScope::execute(command, request.str(), bytes);
        size_t offset = 0;
        for (const auto &output : generic->getOutputs()) {
            std::memcpy(output->getRawDataPtr<void *>(),
                        response.data() + offset, output->getBytes());
            offset += output->getBytes();
        }
        fallback.finish();
    }
    PerfRecord tune(const Operator &op,
                    const RuntimeObj *context) const override {
        const auto start = std::chrono::steady_clock::now();
        compute(op, context);
        return make_ref<PerfRecordObj>(
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start)
                .count());
    }
};
REGISTER_KERNEL(Device::CPU, OpType::LSMSequence, RVNESequenceCpu,
                "LSMSequence_CPU_QEMU");
REGISTER_KERNEL(Device::CPU, OpType::MLP, RVNESequenceCpu, "MLP_CPU_QEMU");
} // namespace infini
