#include "operators/batch_norm.h"
#include "core/kernel.h"
#include "cuda/cuda_kernel_without_config.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "cuda/cuda_unary.h"
#include <iostream>
#include <vector>

namespace infini {
class BatchNormCudnn : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<BatchNormObj>(_op);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        cudnnStatus_t stat;
        IT_ASSERT(op->getDType() == DataType::Float16 ||
                  op->getDType() == DataType::Float32);
        auto cudnnDataType = cudnnDataTypeConvert(op->getDType());
        void *const inData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const outData = (op->getOutput()->getRawDataPtr<void *>());
        const size_t channels = op->getInputs(1)->size();
        float *workspace = static_cast<float *>(
            context->getWorkspace(4 * channels * sizeof(float)));
        auto parameterAsFloat = [&](size_t index) -> const float * {
            auto parameter = op->getInputs(index);
            if (parameter->getDType() == DataType::Float32)
                return parameter->getRawDataPtr<float *>();
            IT_ASSERT(parameter->getDType() == DataType::Float16);
            float *converted = workspace + (index - 1) * channels;
            cast_kernel<half, float>(parameter->getRawDataPtr<half *>(),
                                     converted, channels);
            return converted;
        };
        const float *meanData = parameterAsFloat(1);
        const float *varData = parameterAsFloat(2);
        const float *scaleData = parameterAsFloat(3);
        const float *biasData = parameterAsFloat(4);

        // Debug: copy data to CPU and print
        // std::vector<float> inDebug(12);
        // std::vector<float> meanDebug(3);
        // std::vector<float> varDebug(3);
        // std::vector<float> ScaleDebug(3);
        // std::vector<float> biasDebug(3);
        // context->copyBlobToCPU(inDebug.data(), inData, 12 * sizeof(float));
        // context->copyBlobToCPU(meanDebug.data(), meanData, 3 * sizeof(float));
        // context->copyBlobToCPU(varDebug.data(), varData, 3 * sizeof(float));
        // context->copyBlobToCPU(ScaleDebug.data(), scaleData, 3 * sizeof(float));
        // context->copyBlobToCPU(biasDebug.data(), biasData, 3 * sizeof(float));
        // std::cout << "Input data: ";
        // for (int i = 0; i < 12; i++) std::cout << inDebug[i] << " ";
        // std::cout << std::endl;
        // std::cout << "Mean data: ";
        // for (int i = 0; i < 3; i++) std::cout << meanDebug[i] << " ";
        // std::cout << std::endl;
        // std::cout << "Var data: ";
        // for (int i = 0; i < 3; i++) std::cout << varDebug[i] << " ";
        // std::cout << std::endl;
        // std::cout << "Scale data: ";
        // for (int i = 0; i < 3; i++) std::cout << ScaleDebug[i] << " ";
        // std::cout << std::endl;
        // std::cout << "Bias data: ";
        // for (int i = 0; i < 3; i++) std::cout << biasDebug[i] << " ";
        // std::cout << std::endl;

        // Only 4D and 5D tensors are supported by
        // cudnnBatchNormalizationForwardInference
        if (auto dims = op->getInputs(0)->getDims(); dims.size() < 4) {
            auto dims_t = dims;
            for (size_t i = dims_t.size(); i < 4; ++i) {
                dims_t.push_back(1);
            }
            op->getInputs(0)->setShape(dims_t);
        }
        auto dims = op->getInputs(0)->getDims();
        IT_ASSERT(dims.size() == 4);

        int dimArray[4], strideArray[4], dimPArray[4], stridePArray[4];
        for (size_t i = 0; i < dims.size(); ++i) {
            dimArray[i] = dims[i];
            strideArray[i] = op->getInputs(0)->getStride()[i];
            dimPArray[i] = 1;
            stridePArray[i] = 1;
        }
        dimPArray[1] = op->getInputs(1)->getDims()[0];
        stridePArray[0] = op->getInputs(1)->getDims()[0];
        // get inputs
        cudnnTensorDescriptor_t inDesc;
        checkCudnnError(cudnnCreateTensorDescriptor(&inDesc));
        checkCudnnError(cudnnSetTensorNdDescriptor(
            inDesc, cudnnDataType, dims.size(), dimArray, strideArray));

        // get bnScaleBiasMeanVarDesc
        cudnnTensorDescriptor_t paraDesc;
        checkCudnnError(cudnnCreateTensorDescriptor(&paraDesc));
        checkCudnnError(cudnnSetTensorNdDescriptor(
            paraDesc, CUDNN_DATA_FLOAT, dims.size(), dimPArray, stridePArray));

        float alpha = 1.f, beta = 0.f;
        // This mode is intended for use after convolutional layers
        stat = cudnnBatchNormalizationForwardInference(
            context->cudnnHandle(), CUDNN_BATCHNORM_SPATIAL, &alpha, &beta,
            inDesc, inData, inDesc, outData, paraDesc, scaleData, biasData,
            meanData, varData, op->getEps());
        checkCudnnError(stat);

        // Destories in CUDA does not require sync. But cuDNN does not state
        // whether sync is required before destories.
        checkCudnnError(cudnnDestroyTensorDescriptor(inDesc));
        checkCudnnError(cudnnDestroyTensorDescriptor(paraDesc));
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::BatchNormalization, BatchNormCudnn,
                "BatchNorm_cuDNN_CUDA");
} // namespace infini
