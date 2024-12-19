#include "operators/topk.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"
#include "bang/bang_topk.h"

namespace infini {
class TopKCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<TopKObj>(_op);
        auto input = op->getInputs(0);

        auto output_0 = op->getOutput(0);
        auto output_1 = op->getOutput(1);
        void *const source = input->getRawDataPtr<void *>();
        void *const Values = output_0->getRawDataPtr<void *>();
        void *const Indices = output_1->getRawDataPtr<void *>();

        int axis = op->getAxis();
        int Largest = op->getLargest();
        int Sorted = op->getSorted();

        auto context = dynamic_cast<const BangRuntimeObj *>(_context);
        auto K = op->getTopk();
        int64_t topk_ = K[0];

        auto aDim = op->getInputs(0)->getDims();
        int dimsize = aDim[axis];
        int othersize = input->size() / dimsize;
        if (op->getOpType() == OpType::TopK) {
            if (op->getDType() == DataType::Float32) {
                TopKUnion_f32(context->cnnlHandle(), (float *)source, topk_,
                              (float *)Values, (int64_t *)Indices, othersize,
                              dimsize, Largest, Sorted);
            } else if (op->getDType() == DataType::Float16) {
                TopKUnion_f16(context->cnnlHandle(), (uint16_t *)source, topk_,
                              (uint16_t *)Values, (int64_t *)Indices, othersize,
                              dimsize, Largest, Sorted);
            }
        } else {
            IT_TODO_HALT();
        }
        // int ndim = aDim.size();
        // std::vector<int> inDim(ndim);
        // std::vector<int> outDim = inDim;
        // for (int i = 0; i < ndim; i++) {
        //     inDim[i] = aDim[i];
        //     outDim[i] = aDim[i];
        // }

        // outDim[axis] = topk_[0];
        // cnnlTensorDescriptor_t aDesc, cDesc, index_desc;
        // checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        // checkCnnlError(cnnlSetTensorDescriptor(
        //     aDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(op->getDType()),
        //     inDim.size(), inDim.data()));
        // checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        // checkCnnlError(cnnlSetTensorDescriptor(
        //     cDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(op->getDType()),
        //     outDim.size(), outDim.data()));
        // checkCnnlError(cnnlCreateTensorDescriptor(&index_desc));
        // checkCnnlError(cnnlSetTensorDescriptor(index_desc, CNNL_LAYOUT_ARRAY,
        //                                        CNNL_DTYPE_INT32,
        //                                        outDim.size(),
        //                                        outDim.data()));
        // const bool largest = (Largest > 0 ? true : false);
        // const bool sorted = (Sorted > 0 ? true : false);
        // cnnlTopKTensor(context->cnnlHandle(), aDesc, source, topk_[0], axis,
        //                largest, sorted, cDesc, Values, index_desc, Indices);

        // checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        // checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
        // checkCnnlError(cnnlDestroyTensorDescriptor(index_desc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::TopK, TopKCnnl, "TopK_CNNL");
}; // namespace infini
