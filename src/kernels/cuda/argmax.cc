#include "cuda/cuda_argmax.h"
#include "cuda/cuda_runtime.h"
#include "operators/argmax.h"
#include "cuda/cuda_kernel_wihtout_config.h"

namespace infini {

class ArgMaxCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ArgMaxObj>(_op);
        std::cout << "ArgMaxCuda" << std::endl;  
        Tensor input = op->getInputs(0);
        DataType dtype = input->getDType();
        std::cout << "Input DataType is " << input->getDType().toString() << std::endl;
        void * const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
        int64_t * const outputData = (op->getOutput()->getRawDataPtr<int64_t *>());
        const auto &shapeInput = input->getDims(); // input shape
        for(auto a : shapeInput)
            std::cout <<"a is " << a << " ";
        std::cout << "shapeInput size is " << shapeInput.size() << std::endl;
        std::cout << std::endl;
        std::cout << "Axis is " << op->getAxis() << std::endl;
        int a = 0;
        argmax_kernel(
            inputData,
            outputData,
            shapeInput.data(),
            shapeInput.size(),
            op->getAxis(),
            op->getKeepDims(),
            op->getSelectLastIndex(),
            dtype
        );
        std::cout << "ArgMaxCuda" << std::endl;
        if(outputData == nullptr)
            std::cout << "outputData[0] is NULL" << std::endl;
        else
            std::cout << outputData[0]<< std::endl;

    }
};

REGISTER_KERNEL(Device::CUDA, OpType::ArgMax, ArgMaxCuda, "ArgMax_CUDA");

} // namespace infini
