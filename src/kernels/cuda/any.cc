#include "operators/any.h"
#include "cuda/cuda_any.h"
#include "cuda/cuda_conv2dreduce.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"

namespace infini {

class AnyCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<AnyObj>(_op);

        auto inputs = op->getInputs();
        auto outputs = op->getOutputs();

        vector<float *> inputsRawPtr;
        for (auto &input : inputs) {
            inputsRawPtr.emplace_back(input->getRawDataPtr<float *>());
        }
        vector<float *> outputsRawPtr;
        for (auto &output : outputs) {
            outputsRawPtr.emplace_back(output->getRawDataPtr<float *>());
        }

        any_kernel_mapping(inputsRawPtr, outputsRawPtr, op->getKernelName(),
                           op->getOpAttrVector());
    }
};

void any_kernel_mapping(vector<float *> inputs, vector<float *> outputs,
                        const string &kernelName, const vector<int> &attr) {
    if (kernelName == "conv2dreduce_kernel") {
        IT_ASSERT(attr.size() == 15);
        IT_ASSERT(inputs.size() == 1 || inputs.size() == 2)
        IT_ASSERT(outputs.size() == 1);
        conv2dreduce_kernel(inputs[0], inputs.size() > 1 ? inputs[1] : nullptr,
                            outputs[0], attr[0] != 0, attr[1], attr[2], attr[3],
                            attr[4], attr[5], attr[6], attr[7], attr[8],
                            attr[9], attr[10], attr[11], attr[12], attr[13],
                            attr[14]);
    } else {
        std::cout << "Unimplemented AnyOp cuda kernel: " << kernelName
                  << std::endl;
        IT_TODO_HALT();
    }
}

REGISTER_KERNEL(Device::CUDA, OpType::Any, DataType::Float32, AnyCuda,
                "Any_CUDA_Float32");

} // namespace infini
