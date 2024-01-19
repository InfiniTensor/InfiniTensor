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

        auto input_num = op->numInputs();

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *biasData = NULL;
        if (input_num > 2) {
            biasData = (op->getInputs(2)->getRawDataPtr<void *>());
        }
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, bDesc, cDesc, biasDesc;
        auto dimInputs0 = op->getInputs(0)->getDims();
        auto dimInputs1 = op->getInputs(1)->getDims();
        std::vector<int> dimBias;
        if (input_num > 2) {
            dimBias = op->getInputs(2)->getDims();
        }

        auto dimOutput = op->getOutput()->getDims();

        float alpha = 1.0;
        float beta = 0.0;

        int32_t transA = op->getTransA();
        int32_t transB = op->getTransB();

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(
            cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(op->getDType()),
                                    dimInputs0.size(), dimInputs0.data()));

        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(
            cnnlSetTensorDescriptor(bDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(op->getDType()),
                                    dimInputs1.size(), dimInputs1.data()));

        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(
            cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(op->getDType()),
                                    dimOutput.size(), dimOutput.data()));

        if (input_num > 2) {
            checkCnnlError(cnnlCreateTensorDescriptor(&biasDesc));
            checkCnnlError(cnnlSetTensorDescriptor(
                biasDesc, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(op->getDType()), dimBias.size(),
                dimBias.data()));
        }

        cnnlMatMulDescriptor_t bmm_desc;
        cnnlMatMulDescCreate(&bmm_desc);
        cnnlSetMatMulDescAttr(bmm_desc, CNNL_MATMUL_DESC_TRANSA, &transA,
                              sizeof(int32_t));
        cnnlSetMatMulDescAttr(bmm_desc, CNNL_MATMUL_DESC_TRANSB, &transB,
                              sizeof(int32_t));

        cnnlMatMulAlgo_t bmm_algo;
        cnnlMatMulAlgoCreate(&bmm_algo);

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

        wsData = NULL;
        if (input_num > 2) {
            cnnlGetBiasAddWorkspaceSize(context->cnnlHandle(), biasDesc, cDesc,
                                        &wsSize);
            stat = cnnlBiasAdd(context->cnnlHandle(), &alpha, biasDesc,
                               biasData, wsData, wsSize, &alpha, cDesc, cData);
            if (stat != CNNL_STATUS_SUCCESS)
                return;
        }

        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(bDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
        if (input_num > 2) {
            checkCnnlError(cnnlDestroyTensorDescriptor(biasDesc));
        }
        checkCnnlError(cnnlMatMulDescDestroy(bmm_desc));
        checkCnnlError(cnnlMatMulAlgoDestroy(bmm_algo));
        checkCnnlError(cnnlDestroyMatMulHeuristicResult(desc));
    }
};

REGISTER_KERNEL(Device::BANG, OpType::MatMul, MatmulCnnl, "Matmul_cnnl_BANG");
}; // namespace infini
