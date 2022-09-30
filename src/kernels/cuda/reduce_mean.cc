#include "operators/reduce_mean.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"

namespace infini {
class ReduceMeanCudnn : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ReduceMeanObj>(_op);
        auto input = op->getInputs(0);
        auto output = op->getOutput();
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);

        // Each dimension of the output tensor C must match the corresponding
        // dimension of the input tensor A or must be equal to 1. The dimensions
        // equal to 1 indicate the dimensions of A to be reduced.
        int nInDims = input->getDims().size();
        IT_ASSERT(CUDNN_DIM_MAX >= nInDims);
        int inDimArray[CUDNN_DIM_MAX], outDimArray[CUDNN_DIM_MAX],
            inStrideArray[CUDNN_DIM_MAX], outStrideArray[CUDNN_DIM_MAX];
        for (int i = 0; i < nInDims; ++i) {
            inDimArray[i] = input->getDims()[i];
            inStrideArray[i] = input->getStride()[i];
        }
        Shape d = output->getDims();
        if (!op->getKeepDims()) {
            d = input->getDims();
            for (size_t i = 0; i < d.size(); ++i)
                if (op->isReduced(i))
                    d[i] = 1;
        }
        int stride = 1;
        for (int i = nInDims - 1; i >= 0; --i) {
            outDimArray[i] = d[i];
            outStrideArray[i] = stride;
            stride *= d[i];
        }

        // cudnnSetTensorNdDescriptor is used when nDim>3, otherwise,it is
        // recomended to use cudnnSetTensor4dDescriptor and set the unused
        // dimension size to 1.
        // get inputs outputs
        cudnnTensorDescriptor_t inDesc;
        checkCudnnError(cudnnCreateTensorDescriptor(&inDesc));
        cudnnTensorDescriptor_t outDesc;
        checkCudnnError(cudnnCreateTensorDescriptor(&outDesc));
        if (nInDims > 3) {
            checkCudnnError(cudnnSetTensorNdDescriptor(
                inDesc, CUDNN_DATA_FLOAT, nInDims, inDimArray, inStrideArray));
            checkCudnnError(
                cudnnSetTensorNdDescriptor(outDesc, CUDNN_DATA_FLOAT, nInDims,
                                           outDimArray, outStrideArray));
        } else {
            int idims[4] = {1, 1, 1, 1}, odims[4] = {1, 1, 1, 1};
            for (int i = 0; i < nInDims; ++i) {
                idims[4 - i - 1] = input->getDims()[nInDims - i - 1];
            }
            for (int i = 0; i < nInDims; ++i) {
                odims[4 - i - 1] = d[nInDims - i - 1];
            }

            checkCudnnError(cudnnSetTensor4dDescriptor(
                inDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, idims[0], idims[1],
                idims[2], idims[3]));
            checkCudnnError(cudnnSetTensor4dDescriptor(
                outDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, odims[0],
                odims[1], odims[2], odims[3]));
        }

        // get reduce descriptor
        cudnnReduceTensorDescriptor_t reduceDesc;
        checkCudnnError(cudnnCreateReduceTensorDescriptor(&reduceDesc));
        checkCudnnError(cudnnSetReduceTensorDescriptor(
            reduceDesc, CUDNN_REDUCE_TENSOR_AVG, CUDNN_DATA_FLOAT,
            CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES,
            CUDNN_32BIT_INDICES));

        // get workspace
        size_t workspaceSize = 0;
        checkCudnnError(
            cudnnGetReductionWorkspaceSize(context->cudnnHandle(), reduceDesc,
                                           inDesc, outDesc, &workspaceSize));
        CudaPtr wsData = context->getWorkspace(workspaceSize);

        // get index workspace
        size_t idxWorkspaceSize = 0;
        checkCudnnError(
            cudnnGetReductionIndicesSize(context->cudnnHandle(), reduceDesc,
                                         inDesc, outDesc, &idxWorkspaceSize));
        CudaPtr idxWsData = context->getWorkspace(idxWorkspaceSize);

        // reduce
        float alpha = 1.f, beta = 0.f;
        void *const inData = (input->getRawDataPtr<void *>());
        void *const outData = (output->getRawDataPtr<void *>());
        checkCudnnError(cudnnReduceTensor(context->cudnnHandle(), reduceDesc,
                                          idxWsData, idxWorkspaceSize, wsData,
                                          workspaceSize, &alpha, inDesc, inData,
                                          &beta, outDesc, outData));

        // Destories in CUDA does not require sync. But cuDNN does not state
        // whether sync is required before destories.
        checkCudnnError(cudnnDestroyTensorDescriptor(inDesc));
        checkCudnnError(cudnnDestroyTensorDescriptor(outDesc));
        checkCudnnError(cudnnDestroyReduceTensorDescriptor(reduceDesc));
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::ReduceMean, DataType::Float32,
                ReduceMeanCudnn, "ReduceMean_cuDNN_CUDA_Float32");
}; // namespace infini
