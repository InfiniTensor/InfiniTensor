#include "operators/unary.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"

namespace infini {
class LrnCudnn : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<LrnObj>(_op);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        int lrn_n = op->getFeatureNum();
        float alpha = op->getAlpha();
        float beta = op->getBeta();
        float bias = op->getBias();

        cudnnTensorDescriptor_t aDesc, cDesc;
        auto dim = op->getOutput()->getDims();
        if (dim.size() != 4)
            IT_TODO_HALT();

        checkCudnnError(cudnnCreateTensorDescriptor(&aDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(aDesc, CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_FLOAT, dim[0], dim[1], dim[2], dim[3]));
        checkCudnnError(cudnnCreateTensorDescriptor(&cDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(cDesc, CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_FLOAT, dim[0], dim[1], dim[2], dim[3]));

        cudnnLRNDescriptor_t lrn_desc;
        checkCudnnError(cudnnCreateLRNDescriptor(&lrn_desc));
        checkCudnnError(cudnnSetLRNDescriptor(lrn_desc, (unsigned int)lrn_n, (double)alpha, double(beta), double(bias)));
        cudnnStatus_t stat = cudnnLRNCrossChannelForward(context->cudnnHandle(), lrn_desc, CUDNN_LRN_CROSS_CHANNEL_DIM1, &alpha, aDesc, aData,
                                                         &beta, cDesc, cData);
        if (stat != CUDNN_STATUS_SUCCESS)
            return;

        // Destories in CUDA does not require sync. But cudnn does not state
        // whether sync is required before destories.
        checkCudnnError(cudnnDestroyLRNDescriptor(lrn_desc));
        checkCudnnError(cudnnDestroyTensorDescriptor(aDesc));
        checkCudnnError(cudnnDestroyTensorDescriptor(cDesc));
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Lrn, DataType::Float32,
                LrnCudnn, "Lrn_cudnn_CUDA_Float32");

}; // namespace infini
