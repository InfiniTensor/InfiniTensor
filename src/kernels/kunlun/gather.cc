#include "operators/gather.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"

namespace infini {
class GatherXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<GatherObj>(_op);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto aShape = op->getInputs(0)->getDims();
        auto bSize = op->getInputs(1)->size();
        auto axis = op->getAxis();

        std::cout << "Shape of aData: "
                  << vecToString<int>(op->getInputs(0)->getDims()) << std::endl;
        Tensor aDataCpu =
            op->getInputs(0)->clone(NativeCpuRuntimeObj::getInstance());
        aDataCpu->printData();

        std::vector<float> input0(op->getInputs(0)->size(), 0.0f);
        xpu_memcpy((void *)&(input0[0]), aData,
                   op->getInputs(0)->size() * sizeof(float),
                   XPUMemcpyKind::XPU_DEVICE_TO_HOST);
        xpu_wait();
        std::cout << "Raw data of aData: " << vecToString<float>(input0)
                  << std::endl;
        std::cout << std::endl;

        std::cout << "Shape of bData: "
                  << vecToString<int>(op->getInputs(1)->getDims()) << std::endl;
        Tensor bDataCpu =
            op->getInputs(1)->clone(NativeCpuRuntimeObj::getInstance());
        bDataCpu->printData();

        std::vector<int> input1(op->getInputs(1)->size(), 0);
        xpu_memcpy((void *)&(input1[0]), bData,
                   op->getInputs(1)->size() * sizeof(float),
                   XPUMemcpyKind::XPU_DEVICE_TO_HOST);
        xpu_wait();
        std::cout << "Raw data of bData: " << vecToString<int>(input1)
                  << std::endl;
        std::cout << std::endl;

        auto ret = baidu::xpu::api::gather<float, int>(
            context->KUNLUNHandle(), (float *)aData, (int *)bData,
            (float *)cData, aShape, bSize, axis);
        xpu_wait();
        std::cout << "Shape of cData: "
                  << vecToString<int>(op->getOutput()->getDims()) << std::endl;
        Tensor cDataCpu =
            op->getOutput()->clone(NativeCpuRuntimeObj::getInstance());
        cDataCpu->printData();

        std::vector<float> output(op->getOutput()->size(), 0.0f);
        xpu_memcpy((void *)&(output[0]), cData,
                   op->getOutput()->size() * sizeof(float),
                   XPUMemcpyKind::XPU_DEVICE_TO_HOST);
        xpu_wait();
        std::cout << "Raw data of cData: " << vecToString<float>(output)
                  << std::endl;
        std::cout << std::endl;

        assert(ret == 0);
        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::Gather, DataType::Float32, GatherXdnn,
                "Gather_xdnn_KUNLUN_Float32");
}; // namespace infini
