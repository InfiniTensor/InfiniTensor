#include "operators/reduce_mean.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"

namespace infini {
class ReduceMeanCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ReduceMeanObj>(_op);
        auto input = op->getInputs(0);
        auto output = op->getOutput();
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        // Each dimension of the output tensor C must match the corresponding
        // dimension of the input tensor A or must be equal to 1. The dimensions
        // equal to 1 indicate the dimensions of A to be reduced.
        int nInDims = input->getRank();
        IT_ASSERT(CNNL_DIM_MAX >= nInDims);
        int inDimArray[CNNL_DIM_MAX], outDimArray[CNNL_DIM_MAX],
            inStrideArray[CNNL_DIM_MAX], outStrideArray[CNNL_DIM_MAX];
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
        cnnlTensorDescriptor_t inDesc;
        checkCnnlError(cnnlCreateTensorDescriptor(&inDesc));
        cnnlTensorDescriptor_t outDesc;
        checkCnnlError(cnnlCreateTensorDescriptor(&outDesc));
        if (nInDims > 3) {
            checkCnnlError(cnnlSetTensorDescriptorEx(
                inDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, nInDims,
                inDimArray, inStrideArray));
            checkCnnlError(cnnlSetTensorDescriptorEx(
                outDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT, nInDims,
                outDimArray, outStrideArray));
        } else {
            int idims[4] = {1, 1, 1, 1}, odims[4] = {1, 1, 1, 1};
            for (int i = 0; i < nInDims; ++i) {
                idims[4 - i - 1] = input->getDims()[nInDims - i - 1];
            }
            for (int i = 0; i < nInDims; ++i) {
                odims[4 - i - 1] = d[nInDims - i - 1];
            }

            checkCnnlError(cnnlSetTensorDescriptor(inDesc, CNNL_LAYOUT_ARRAY,
                                                   CNNL_DTYPE_FLOAT, 4, idims));
            checkCnnlError(cnnlSetTensorDescriptor(outDesc, CNNL_LAYOUT_ARRAY,
                                                   CNNL_DTYPE_FLOAT, 4, odims));
        }

        auto axes_set = op->getAxes();

        std::vector<int> axes;
        axes.assign(axes_set.begin(), axes_set.end());

        // get reduce descriptor
        cnnlReduceDescriptor_t reduceDesc;
        checkCnnlError(cnnlCreateReduceDescriptor(&reduceDesc));
        checkCnnlError(cnnlSetReduceDescriptor_v2(
            reduceDesc, axes.data(), axes.size(), CNNL_REDUCE_AVG,
            CNNL_DTYPE_FLOAT, CNNL_NOT_PROPAGATE_NAN, CNNL_REDUCE_ONLY_INDICES,
            CNNL_32BIT_INDICES, 0.0));

        // get workspace
        size_t workspaceSize = 0;
        checkCnnlError(cnnlGetReduceOpWorkspaceSize(context->cnnlHandle(),
                                                    inDesc, outDesc, reduceDesc,
                                                    &workspaceSize));
        int indicesSize = axes.size() * sizeof(int);
        BangPtr wsData = context->getWorkspace(workspaceSize + indicesSize);

        BangPtr indicesData = (char *)wsData + workspaceSize;
        context->copyBlobFromCPU(indicesData, axes.data(), indicesSize);

        // reduce
        float alpha = 1.f, beta = 0.f;
        void *const inData = (input->getRawDataPtr<void *>());
        void *const outData = (output->getRawDataPtr<void *>());
        checkCnnlError(cnnlReduce(
            context->cnnlHandle(), reduceDesc, wsData, workspaceSize, &alpha,
            inDesc, inData, indicesSize, indicesData, &beta, outDesc, outData));

        // Destories in CUDA does not require sync. But cuDNN does not state
        // whether sync is required before destories.
        checkCnnlError(cnnlDestroyTensorDescriptor(inDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(outDesc));
        checkCnnlError(cnnlDestroyReduceDescriptor(reduceDesc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::ReduceMean, DataType::Float32,
                ReduceMeanCnnl, "ReduceMean_cnnl_BANG_Float32");

}; // namespace infini
