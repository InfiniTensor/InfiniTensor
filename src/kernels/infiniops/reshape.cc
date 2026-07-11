#include "infiniops_common.h"
#include "operators/reshape.h"
#include "operators/squeeze.h"
#include "operators/unsqueeze.h"

namespace infini {
namespace {

class CopyInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        context->copyBlob(_op->getOutput().get(), _op->getInputs(0).get());
    }
};

} // namespace

#ifdef USE_CUDA
REGISTER_KERNEL(Device::CUDA, OpType::Reshape, CopyInfiniOps,
                "Reshape_InfiniOps_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Flatten, CopyInfiniOps,
                "Flatten_InfiniOps_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Identity, CopyInfiniOps,
                "Identity_InfiniOps_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Squeeze, CopyInfiniOps,
                "Squeeze_InfiniOps_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Unsqueeze, CopyInfiniOps,
                "Unsqueeze_InfiniOps_CUDA");
#endif
#ifdef USE_BANG
REGISTER_KERNEL(Device::BANG, OpType::Reshape, CopyInfiniOps, "Reshape_Runtime_BANG");
REGISTER_KERNEL(Device::BANG, OpType::Flatten, CopyInfiniOps, "Flatten_Runtime_BANG");
REGISTER_KERNEL(Device::BANG, OpType::Identity, CopyInfiniOps, "Identity_Runtime_BANG");
REGISTER_KERNEL(Device::BANG, OpType::Squeeze, CopyInfiniOps, "Squeeze_Runtime_BANG");
REGISTER_KERNEL(Device::BANG, OpType::Unsqueeze, CopyInfiniOps, "Unsqueeze_Runtime_BANG");
#endif
#ifdef USE_INTELCPU
REGISTER_KERNEL(Device::INTELCPU, OpType::Reshape, CopyInfiniOps, "Reshape_Runtime_INTELCPU");
REGISTER_KERNEL(Device::INTELCPU, OpType::Flatten, CopyInfiniOps, "Flatten_Runtime_INTELCPU");
REGISTER_KERNEL(Device::INTELCPU, OpType::Identity, CopyInfiniOps, "Identity_Runtime_INTELCPU");
#endif
#ifdef USE_KUNLUN
REGISTER_KERNEL(Device::KUNLUN, OpType::Reshape, CopyInfiniOps, "Reshape_Runtime_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::Flatten, CopyInfiniOps, "Flatten_Runtime_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::Identity, CopyInfiniOps, "Identity_Runtime_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::Squeeze, CopyInfiniOps, "Squeeze_Runtime_KUNLUN");
#endif
#ifdef USE_ASCEND
REGISTER_KERNEL(Device::ASCEND, OpType::Reshape, CopyInfiniOps, "Reshape_Runtime_ASCEND");
REGISTER_KERNEL(Device::ASCEND, OpType::Flatten, CopyInfiniOps, "Flatten_Runtime_ASCEND");
REGISTER_KERNEL(Device::ASCEND, OpType::Identity, CopyInfiniOps, "Identity_Runtime_ASCEND");
REGISTER_KERNEL(Device::ASCEND, OpType::Squeeze, CopyInfiniOps, "Squeeze_Runtime_ASCEND");
REGISTER_KERNEL(Device::ASCEND, OpType::Unsqueeze, CopyInfiniOps, "Unsqueeze_Runtime_ASCEND");
#endif
#ifdef USE_ILUVATAR
REGISTER_KERNEL(Device::ILUVATAR, OpType::Reshape, CopyInfiniOps, "Reshape_Runtime_ILUVATAR");
REGISTER_KERNEL(Device::ILUVATAR, OpType::Flatten, CopyInfiniOps, "Flatten_Runtime_ILUVATAR");
REGISTER_KERNEL(Device::ILUVATAR, OpType::Identity, CopyInfiniOps, "Identity_Runtime_ILUVATAR");
REGISTER_KERNEL(Device::ILUVATAR, OpType::Squeeze, CopyInfiniOps, "Squeeze_Runtime_ILUVATAR");
REGISTER_KERNEL(Device::ILUVATAR, OpType::Unsqueeze, CopyInfiniOps, "Unsqueeze_Runtime_ILUVATAR");
#endif
#ifdef USE_METAX
REGISTER_KERNEL(Device::METAX, OpType::Reshape, CopyInfiniOps, "Reshape_Runtime_METAX");
REGISTER_KERNEL(Device::METAX, OpType::Flatten, CopyInfiniOps, "Flatten_Runtime_METAX");
REGISTER_KERNEL(Device::METAX, OpType::Identity, CopyInfiniOps, "Identity_Runtime_METAX");
REGISTER_KERNEL(Device::METAX, OpType::Squeeze, CopyInfiniOps, "Squeeze_Runtime_METAX");
REGISTER_KERNEL(Device::METAX, OpType::Unsqueeze, CopyInfiniOps, "Unsqueeze_Runtime_METAX");
#endif
#ifdef USE_MOORE
REGISTER_KERNEL(Device::MOORE, OpType::Reshape, CopyInfiniOps, "Reshape_Runtime_MOORE");
REGISTER_KERNEL(Device::MOORE, OpType::Flatten, CopyInfiniOps, "Flatten_Runtime_MOORE");
REGISTER_KERNEL(Device::MOORE, OpType::Identity, CopyInfiniOps, "Identity_Runtime_MOORE");
REGISTER_KERNEL(Device::MOORE, OpType::Squeeze, CopyInfiniOps, "Squeeze_Runtime_MOORE");
REGISTER_KERNEL(Device::MOORE, OpType::Unsqueeze, CopyInfiniOps, "Unsqueeze_Runtime_MOORE");
#endif
#ifdef USE_HYGON
REGISTER_KERNEL(Device::HYGON, OpType::Reshape, CopyInfiniOps, "Reshape_Runtime_HYGON");
REGISTER_KERNEL(Device::HYGON, OpType::Flatten, CopyInfiniOps, "Flatten_Runtime_HYGON");
REGISTER_KERNEL(Device::HYGON, OpType::Identity, CopyInfiniOps, "Identity_Runtime_HYGON");
REGISTER_KERNEL(Device::HYGON, OpType::Squeeze, CopyInfiniOps, "Squeeze_Runtime_HYGON");
REGISTER_KERNEL(Device::HYGON, OpType::Unsqueeze, CopyInfiniOps, "Unsqueeze_Runtime_HYGON");
#endif

} // namespace infini
