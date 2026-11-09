#include "operators/motion.h"
#include "core/kernel.h"
#include "utils/motion_compute.h"
#include "utils/rvne_session.h"
#include <cstring>
#include <sstream>
namespace infini {
class MotionCpu : public CpuKernelWithoutConfig {
    void compute(const Operator &generic, const RuntimeObj *) const override {
        auto op = as<MotionObj>(generic);
        int s = op->getInputs(0)->getDims()[0];
        auto d = [&](int i) {
            return op->getInputs(i)->getRawDataPtr<double *>();
        };
        auto f = [&](int i) {
            return op->getInputs(i)->getRawDataPtr<float *>();
        };
        if (!op->executesQemu()) {
            if (op->getOpType() == OpType::MotionFeatures)
                motion::features(s, d(0), d(1), d(2), d(3), d(4), d(5), d(6),
                                 d(7),
                                 op->getOutput(0)->getRawDataPtr<float *>(),
                                 op->getOutput(1)->getRawDataPtr<double *>(),
                                 op->getOutput(2)->getRawDataPtr<double *>(),
                                 op->getOutput(3)->getRawDataPtr<int32_t *>());
            else
                motion::readout(s, f(0), f(1), f(2), f(3), f(4), d(5), d(6),
                                op->getOutput()->getRawDataPtr<double *>());
            return;
        }
        std::ostringstream payload(std::ios::binary);
        uint32_t samples = s;
        rvne::write(payload, &samples, 1);
        for (auto &input : op->getInputs())
            rvne::write(payload, input->getRawDataPtr<char *>(),
                        input->getBytes());
        size_t bytes = 0;
        for (auto &output : op->getOutputs())
            bytes += output->getBytes();
        rvne::SessionScope scope;
        auto result = rvne::SessionScope::execute(
            op->getOpType() == OpType::MotionFeatures ? rvne::MotionFeatures
                                                      : rvne::MotionReadout,
            payload.str(), bytes);
        size_t offset = 0;
        for (auto &output : op->getOutputs()) {
            std::memcpy(output->getRawDataPtr<void *>(), result.data() + offset,
                        output->getBytes());
            offset += output->getBytes();
        }
        scope.finish();
    }
};
REGISTER_KERNEL(Device::CPU, OpType::MotionFeatures, MotionCpu,
                "MotionFeatures_CPU_QEMU");
REGISTER_KERNEL(Device::CPU, OpType::MotionReadout, MotionCpu,
                "MotionReadout_CPU_QEMU");
} // namespace infini
