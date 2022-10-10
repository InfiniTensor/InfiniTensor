#include "core/layer.h"

namespace infini {
    Convolution::Convolution(Tensor input_,
                             int pad,
                             int window,
                             int stride,
                             int num) {
        Runtime cpuRuntime = CpuRuntimeObj::getInstance();
        // layout NCHW
        input = input_;
        Shape inputShape = input_->getDims();
        int inputN = inputShape[0];
        int inputC = inputShape[1];
        int inputH = inputShape[2];
        int inputW = inputShape[3];
        Shape weightShape = {num, inputC, window, window};
        weight = make_ref<TensorObj>(weightShape, DataType::Float32, cpuRuntime);
        weight->dataMalloc();
        int outputN = inputN;
        int outputC = num;
        int outputH =  (inputH + 2 * pad - window) / stride + 1;
        int outputW =  (inputW + 2 * pad - window) / stride + 1;
        Shape outputShape = {outputN, outputC, outputH, outputW};
        output = make_ref<TensorObj>(outputShape, DataType::Float32, cpuRuntime);
        output->dataMalloc();
        // backward
        dInput = make_ref<TensorObj>(inputShape, DataType::Float32, cpuRuntime);
        dInput->dataMalloc();
        dWeight = make_ref<TensorObj>(weightShape, DataType::Float32, cpuRuntime);
        dWeight->dataMalloc();
        dOutput = make_ref<TensorObj>(outputShape, DataType::Float32, cpuRuntime);
        dOutput->dataMalloc();
    }

    Tensor Convolution::forward() {
        return output;
    }

    Tensor Convolution::backward() {
        return dInput;
    }

} // namespace infini
