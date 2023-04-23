#include "operators/conv2dreduce.h"
#include "cuda/cuda_conv2dreduce.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"

namespace infini {

class Conv2dReduceCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op, const RuntimeObj *_context) const {
        auto op = as<Conv2dReduceBase>(_op);
        float *const input = (op->getInputs(0)->getRawDataPtr<float *>());
        float *const bias =
            op->getBias() ? (op->getBias()->getRawDataPtr<float *>()) : nullptr;
        float *const output = (op->getOutput()->getRawDataPtr<float *>());

        auto dim = op->getInputs(0)->getDims();
        int n = dim[0], h = dim[1], w = dim[2], f = dim[3], r = dim[4],
            s = dim[5];
        int dh = op->getDh(), dw = op->getDw();
        int sh = op->getSh(), sw = op->getSw();
        int ph = op->getPh(), pw = op->getPw();
        auto odim = op->getOutput()->getDims();
        int oh = odim[1], ow = odim[2];
        bool PReLU = op->getPReLU();
        // float paramReLU = op->getParamReLU();

        auto opType = op->getOpType();

        if (opType == OpType::Conv2dReduce) {
            conv2dreduce_kernel(input, bias, output, PReLU, n, h, w, f, r, s,
                                oh, ow, ph, pw, sh, sw, dh, dw);
        } else {
            convTranspose2dreduce_kernel(input, bias, output, PReLU, n, h, w, f,
                                         r, s, oh, ow, ph, pw, sh, sw, dh, dw);
        }
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Conv2dReduce, DataType::Float32,
                Conv2dReduceCuda, "Conv2dReduce_CUDA_Float32");
REGISTER_KERNEL(Device::CUDA, OpType::Conv2dReduceTranspose, DataType::Float32,
                Conv2dReduceCuda, "Conv2dReduceTranspose_CUDA_Float32");

} // namespace infini
