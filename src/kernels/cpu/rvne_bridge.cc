#include "core/kernel.h"
#include "operators/rvne_ops.h"
#include "utils/mlp_compute.h"
#include "utils/rvne_session.h"
#include <chrono>
#include <cstring>
#include <sstream>

namespace infini {
// The host validates tensors and submits either operation to the same session.
// Only the explicit reference branches below compute on the host.
class RVNEBridgeCpu : public CpuKernelWithoutConfig {
    void compute(const Operator &genericOperator,
                 const RuntimeObj *) const override {
        rvne::Command command;
        std::ostringstream requestPayload(std::ios::binary);
        if (genericOperator->getOpType() == OpType::LSMSequence) {
            auto op = as<LSMSequenceObj>(genericOperator);
            int steps = op->getInputs(0)->getDims()[0];
            int channels = op->getInputs(0)->getDims()[1];
            int neurons = op->getInputs(2)->getDims()[0];
            auto *spikes = op->getInputs(0)->getRawDataPtr<int32_t *>();
            auto *inputWeights = op->getInputs(1)->getRawDataPtr<int32_t *>();
            auto *recurrentWeights =
                op->getInputs(2)->getRawDataPtr<int32_t *>();
            auto *stateIn = op->getInputs(3)->getRawDataPtr<int32_t *>();
            const auto &parameters = op->getParameters();
            lsm::validateInputs(steps, channels, neurons, parameters, spikes,
                                inputWeights, recurrentWeights, stateIn);
            if (!op->executesQemu()) {
                lsm::runReference(steps, channels, neurons, parameters, spikes,
                                  inputWeights, recurrentWeights, stateIn,
                                  op->tracesAllSteps(),
                                  op->getOutput(0)->getRawDataPtr<float *>(),
                                  op->getOutput(1)->getRawDataPtr<int32_t *>(),
                                  op->getOutput(2)->getRawDataPtr<int32_t *>());
                return;
            }
            command = rvne::LSMSequence;
            uint32_t dimensions[4] = {uint32_t(steps), uint32_t(channels),
                                      uint32_t(neurons),
                                      uint32_t(op->tracesAllSteps())};
            rvne::write(requestPayload, dimensions, 4);
            rvne::write(requestPayload, parameters.data(), parameters.size());
        } else {
            auto op = as<MLPObj>(genericOperator);
            const auto inputShape = op->getInputs(0)->getDims();
            uint32_t dimensions[4] = {uint32_t(inputShape[0]),
                                      uint32_t(inputShape[1]),
                                      uint32_t(op->getInputs(1)->getDims()[1]),
                                      uint32_t(op->getInputs(3)->getDims()[1])};
            mlp::validateDimensions(dimensions[0], dimensions[1], dimensions[2],
                                    dimensions[3]);
            for (const auto &input : op->getInputs())
                mlp::validateFinite(input->getRawDataPtr<float *>(),
                                    input->size());
            if (!op->executesQemu()) {
                mlp::mlp_compute(dimensions[0], dimensions[1], dimensions[2],
                                 dimensions[3],
                                 op->getInputs(0)->getRawDataPtr<float *>(),
                                 op->getInputs(1)->getRawDataPtr<float *>(),
                                 op->getInputs(2)->getRawDataPtr<float *>(),
                                 op->getInputs(3)->getRawDataPtr<float *>(),
                                 op->getInputs(4)->getRawDataPtr<float *>(),
                                 op->getOutput()->getRawDataPtr<float *>());
                return;
            }
            command = rvne::MLP;
            rvne::write(requestPayload, dimensions, 4);
        }
        for (const auto &input : genericOperator->getInputs())
            rvne::write(requestPayload, input->getRawDataPtr<char *>(),
                        input->getBytes());
        size_t expectedResponseBytes = 0;
        for (const auto &output : genericOperator->getOutputs())
            expectedResponseBytes += output->getBytes();
        rvne::SessionScope kernelSessionScope;
        auto responsePayload = rvne::SessionScope::execute(
            command, requestPayload.str(), expectedResponseBytes);
        size_t outputOffset = 0;
        for (const auto &output : genericOperator->getOutputs()) {
            std::memcpy(output->getRawDataPtr<void *>(),
                        responsePayload.data() + outputOffset,
                        output->getBytes());
            outputOffset += output->getBytes();
        }
        kernelSessionScope.finish();
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
REGISTER_KERNEL(Device::CPU, OpType::LSMSequence, RVNEBridgeCpu,
                "LSMSequence_CPU_RVNE");
REGISTER_KERNEL(Device::CPU, OpType::MLP, RVNEBridgeCpu, "MLP_CPU_RVNE");
} // namespace infini
