#include "operators/matmul.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"

namespace infini {
class MatmulCnnl : public BangKernelWithoutConfig {
    virtual tuple<float, float> getAlphBeta() const { return {1.f, 0.f}; }
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<MatmulObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, bDesc, cDesc;
        auto dimInputs0 = op->getInputs(0)->getDims();
        auto dimInputs1 = op->getInputs(1)->getDims();
        auto dimOutput = op->getOutput()->getDims();

        int32_t transA = op->getTransA();
        int32_t transB = op->getTransB();

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(
            cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT,
                                    dimInputs0.size(), dimInputs0.data()));

        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(
            cnnlSetTensorDescriptor(bDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT,
                                    dimInputs1.size(), dimInputs1.data()));

        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(
            cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_FLOAT,
                                    dimOutput.size(), dimOutput.data()));

        cnnlMatMulDescriptor_t bmm_desc;
        cnnlMatMulDescCreate(&bmm_desc);
        cnnlSetMatMulDescAttr(bmm_desc, CNNL_MATMUL_DESC_TRANSA, &transA,
                              sizeof(int32_t));
        cnnlSetMatMulDescAttr(bmm_desc, CNNL_MATMUL_DESC_TRANSB, &transB,
                              sizeof(int32_t));

        cnnlMatMulAlgo_t bmm_algo;
        cnnlMatMulAlgoCreate(&bmm_algo);

        float alpha = 1.0;
        float beta = 0.0;
        int count = 0;

        cnnlMatMulHeuristicResult_t desc;
        cnnlCreateMatMulHeuristicResult(&desc);

        cnnlGetBatchMatMulAlgoHeuristic(context->cnnlHandle(), bmm_desc, aDesc,
                                        bDesc, cDesc, NULL, 1, &desc, &count);
        size_t wsSize;
        cnnlGetBatchMatMulHeuristicResult(desc, bmm_algo, &wsSize);
        BangPtr wsData = context->getWorkspace(wsSize);

        cnnlStatus_t stat = cnnlBatchMatMulBCast_v2(
            context->cnnlHandle(), bmm_desc, bmm_algo, &alpha, aDesc, aData,
            bDesc, bData, &beta, cDesc, cData, wsData, wsSize);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(bDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
        checkCnnlError(cnnlMatMulDescDestroy(bmm_desc));
        checkCnnlError(cnnlMatMulAlgoDestroy(bmm_algo));
        checkCnnlError(cnnlDestroyMatMulHeuristicResult(desc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::MatMul, DataType::Float32, MatmulCnnl,
                "Matmul_cnnl_BANG_Float32");
}; // namespace infini
