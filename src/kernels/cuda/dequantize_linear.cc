#include "operators/dequantize_linear.h"
#include "cuda/cuda_dequantize_linear.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"

namespace infini {

class DequantizeLinearCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<DequantizeLinearObj>(_op);

        void *const inputX = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const inputScale = (op->getInputs(1)->getRawDataPtr<void *>());

        void *const output = (op->getOutput()->getRawDataPtr<void *>());

        const int axis = op->getAxis();
        const int stride = op->getInputs(0)->getStride().at(axis);

        auto dims = op->getInputs(0)->getDims();
        int dimsize = dims[op->getAxis()];
        int size = op->getOutput()->size();

        if (op->getInputs(1)->getDType() == DataType::Float32) {

            if (op->numInputs() == 3) {
                void *const inputZeroPoint =
                    (op->getInputs(2)->getRawDataPtr<void *>());

                DequantizeLinearKernel((uint8_t *)inputX, (float *)inputScale,
                                       (float *)output, dimsize, stride,
                                       (uint8_t *)inputZeroPoint, size);
            } else {
                DequantizeLinearKernel((uint8_t *)inputX, (float *)inputScale,
                                       (float *)output, dimsize, stride, size);
            }
        } else if (op->getInputs(1)->getDType() == DataType::Float16) {
            if (op->numInputs() == 3) {
                void *const inputZeroPoint =
                    (op->getInputs(2)->getRawDataPtr<void *>());

                DequantizeLinearKernel((uint8_t *)inputX, (half *)inputScale,
                                       (half *)output, dimsize, stride,
                                       (uint8_t *)inputZeroPoint, size);
            } else {
                DequantizeLinearKernel((uint8_t *)inputX, (half *)inputScale,
                                       (half *)output, dimsize, stride, size);
            }
        }
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::DequantizeLinear, DequantizeLinearCuda,
                "DequantizeLinear_CUDA");

}; // namespace infini
