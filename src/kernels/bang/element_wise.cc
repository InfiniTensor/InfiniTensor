#include "operators/element_wise.h"
#include "bang/bang_kernel_without_config.h"
#include "bang/bang_runtime.h"

namespace infini {
class ElementWiseCnnl : public BangKernelWithoutConfig {
    virtual cnnlOpTensorDesc_t getOpType() const = 0;
    virtual tuple<float, float, float> getAlphBeta() const {
        return {1.f, 1.f, 0.f};
    }
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, bDesc, cDesc;
        auto a_dim = op->getInputs(0)->getDims();
        auto b_dim = op->getInputs(1)->getDims();
        auto c_dim = op->getOutput()->getDims();
        if (a_dim.size() == 0) {
            a_dim.push_back(1);
        }

        if (b_dim.size() == 0) {
            b_dim.push_back(1);
        }

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, a_dim.size(),
                                               a_dim.data()));

        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(bDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, b_dim.size(),
                                               b_dim.data()));

        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, c_dim.size(),
                                               c_dim.data()));

        cnnlOpTensorDescriptor_t opDesc;
        checkCnnlError(cnnlCreateOpTensorDescriptor(&opDesc));
        checkCnnlError(cnnlSetOpTensorDescriptor(
            opDesc, getOpType(), CNNL_DTYPE_FLOAT, CNNL_NOT_PROPAGATE_NAN));

        size_t wsSize;
        cnnlGetOpTensorWorkspaceSize(context->cnnlHandle(), aDesc, bDesc, cDesc,
                                     &wsSize);

        BangPtr wsData = context->getWorkspace(wsSize);

        auto [aAlpha, bAlpha, beta] = getAlphBeta();
        cnnlStatus_t stat = cnnlOpTensor(context->cnnlHandle(), opDesc, &aAlpha,
                                         aDesc, aData, &bAlpha, bDesc, bData,
                                         wsData, wsSize, &beta, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(bDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
        checkCnnlError(cnnlDestroyOpTensorDescriptor(opDesc));
    }
};

class LogicOpCnnl : public BangKernelWithoutConfig {
    virtual cnnlLogicOp_t getOpType() const = 0;
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, bDesc, cDesc;
        auto a_dim = op->getInputs(0)->getDims();
        auto b_dim = op->getInputs(1)->getDims();
        auto c_dim = op->getOutput()->getDims();
        if (a_dim.size() == 0) {
            a_dim.push_back(1);
        }

        if (b_dim.size() == 0) {
            b_dim.push_back(1);
        }

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, a_dim.size(),
                                               a_dim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(bDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, b_dim.size(),
                                               b_dim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, c_dim.size(),
                                               c_dim.data()));

        size_t wsSize;
        cnnlGetLogicOpWorkspaceSize(context->cnnlHandle(), aDesc, bDesc, cDesc,
                                    &wsSize);
        BangPtr wsData = context->getWorkspace(wsSize);

        cnnlStatus_t stat =
            cnnlLogicOp(context->cnnlHandle(), getOpType(), aDesc, aData, bDesc,
                        bData, wsData, wsSize, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(bDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

class BitComputeCnnl : public BangKernelWithoutConfig {
    virtual cnnlBitComputeOp_t getOpType() const = 0;
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, bDesc, cDesc;
        auto a_dim = op->getInputs(0)->getDims();
        auto b_dim = op->getInputs(1)->getDims();
        auto c_dim = op->getOutput()->getDims();
        if (a_dim.size() == 0) {
            a_dim.push_back(1);
        }

        if (b_dim.size() == 0) {
            b_dim.push_back(1);
        }

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_INT32, a_dim.size(),
                                               a_dim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(bDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_INT32, b_dim.size(),
                                               b_dim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_INT32, c_dim.size(),
                                               c_dim.data()));

        size_t wsSize;
        cnnlGetBitComputeWorkspaceSize(context->cnnlHandle(), aDesc, bDesc,
                                       cDesc, &wsSize);
        BangPtr wsData = context->getWorkspace(wsSize);

        cnnlStatus_t stat =
            cnnlBitCompute_v2(context->cnnlHandle(), getOpType(), aDesc, aData,
                              bDesc, bData, cDesc, cData, wsData, wsSize);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(bDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

class DivCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, bDesc, cDesc;
        auto a_dim = op->getInputs(0)->getDims();
        auto b_dim = op->getInputs(1)->getDims();
        auto c_dim = op->getOutput()->getDims();
        if (a_dim.size() == 0) {
            a_dim.push_back(1);
        }

        if (b_dim.size() == 0) {
            b_dim.push_back(1);
        }

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, a_dim.size(),
                                               a_dim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(bDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, b_dim.size(),
                                               b_dim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, c_dim.size(),
                                               c_dim.data()));

        size_t wsSize;
        cnnlGetDivWorkspaceSize(context->cnnlHandle(), aDesc, bDesc, cDesc,
                                &wsSize);
        BangPtr wsData = context->getWorkspace(wsSize);

        cnnlStatus_t stat = cnnlDiv_v2(
            context->cnnlHandle(), CNNL_COMPUTATION_HIGH_PRECISION, aDesc,
            aData, bDesc, bData, wsData, wsSize, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(bDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

class MaximumCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, bDesc, cDesc;
        auto a_dim = op->getInputs(0)->getDims();
        auto b_dim = op->getInputs(1)->getDims();
        auto c_dim = op->getOutput()->getDims();
        if (a_dim.size() == 0) {
            a_dim.push_back(1);
        }

        if (b_dim.size() == 0) {
            b_dim.push_back(1);
        }

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, a_dim.size(),
                                               a_dim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(bDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, b_dim.size(),
                                               b_dim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, c_dim.size(),
                                               c_dim.data()));

        size_t wsSize;
        cnnlGetMaximumWorkspaceSize(context->cnnlHandle(), cDesc, &wsSize);
        BangPtr wsData = context->getWorkspace(wsSize);

        cnnlStatus_t stat =
            cnnlMaximum(context->cnnlHandle(), aDesc, aData, bDesc, bData,
                        cDesc, cData, wsData, wsSize);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(bDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

class MinimumCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, bDesc, cDesc;
        auto a_dim = op->getInputs(0)->getDims();
        auto b_dim = op->getInputs(1)->getDims();
        auto c_dim = op->getOutput()->getDims();
        if (a_dim.size() == 0) {
            a_dim.push_back(1);
        }

        if (b_dim.size() == 0) {
            b_dim.push_back(1);
        }

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, a_dim.size(),
                                               a_dim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(bDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, b_dim.size(),
                                               b_dim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, c_dim.size(),
                                               c_dim.data()));

        size_t wsSize;
        cnnlGetMinimumWorkspaceSize(context->cnnlHandle(), cDesc, &wsSize);
        BangPtr wsData = context->getWorkspace(wsSize);

        cnnlStatus_t stat =
            cnnlMinimum(context->cnnlHandle(), aDesc, aData, bDesc, bData,
                        cDesc, cData, wsData, wsSize);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(bDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

class MSELossCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<MSELossObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        MSELossObj::Reduction reduction = op->getReduction();
        cnnlTensorDescriptor_t aDesc, bDesc, cDesc;
        auto a_dim = op->getInputs(0)->getDims();
        auto b_dim = op->getInputs(1)->getDims();
        auto c_dim = op->getOutput()->getDims();
        if (a_dim.size() == 0) {
            a_dim.push_back(1);
        }

        if (b_dim.size() == 0) {
            b_dim.push_back(1);
        }

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, a_dim.size(),
                                               a_dim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(bDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, b_dim.size(),
                                               b_dim.data()));

        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, c_dim.size(),
                                               c_dim.data()));
        cnnlStatus_t stat;
        if (reduction == MSELossObj::None) {
            stat = cnnlMSELoss(context->cnnlHandle(), CNNL_MSE_LOSS_NONE, aDesc,
                               aData, bDesc, bData, cDesc, cData);
        } else if (reduction == MSELossObj::Sum) {
            stat = cnnlMSELoss(context->cnnlHandle(), CNNL_MSE_LOSS_SUM, aDesc,
                               aData, bDesc, bData, cDesc, cData);
        } else {
            stat = cnnlMSELoss(context->cnnlHandle(), CNNL_MSE_LOSS_MEAN, aDesc,
                               aData, bDesc, bData, cDesc, cData);
        }

        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(bDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

class PowerCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, bDesc, cDesc;
        auto a_dim = op->getInputs(0)->getDims();
        auto b_dim = op->getInputs(1)->getDims();
        auto c_dim = op->getOutput()->getDims();

        if (a_dim.size() == 0) {
            a_dim.push_back(1);
        }

        if (b_dim.size() == 0) {
            b_dim.push_back(1);
        }

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, a_dim.size(),
                                               a_dim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(bDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, b_dim.size(),
                                               b_dim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, c_dim.size(),
                                               c_dim.data()));

        size_t wsSize;
        cnnlGetPowWorkspaceSize(context->cnnlHandle(), aDesc, bDesc, cDesc,
                                &wsSize);
        BangPtr wsData = context->getWorkspace(wsSize);

        cnnlStatus_t stat =
            cnnlPow(context->cnnlHandle(), CNNL_COMPUTATION_HIGH_PRECISION,
                    aDesc, aData, bDesc, bData, wsData, wsSize, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(bDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

class FloorDivCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, bDesc, cDesc;
        auto a_dim = op->getInputs(0)->getDims();
        auto b_dim = op->getInputs(1)->getDims();
        auto c_dim = op->getOutput()->getDims();
        if (a_dim.size() == 0) {
            a_dim.push_back(1);
        }

        if (b_dim.size() == 0) {
            b_dim.push_back(1);
        }

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, a_dim.size(),
                                               a_dim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(bDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, b_dim.size(),
                                               b_dim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, c_dim.size(),
                                               c_dim.data()));

        size_t wsSize;
        cnnlGetFloorDivWorkspaceSize(context->cnnlHandle(), aDesc, bDesc, cDesc,
                                     &wsSize);
        BangPtr wsData = context->getWorkspace(wsSize);

        cnnlStatus_t stat = cnnlFloorDiv_v2(
            context->cnnlHandle(), CNNL_COMPUTATION_HIGH_PRECISION, aDesc,
            aData, bDesc, bData, cDesc, cData, wsData, wsSize);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(bDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

class FloorModCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, bDesc, cDesc;
        auto a_dim = op->getInputs(0)->getDims();
        auto b_dim = op->getInputs(1)->getDims();
        auto c_dim = op->getOutput()->getDims();
        if (a_dim.size() == 0) {
            a_dim.push_back(1);
        }

        if (b_dim.size() == 0) {
            b_dim.push_back(1);
        }

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, a_dim.size(),
                                               a_dim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(bDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, b_dim.size(),
                                               b_dim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, c_dim.size(),
                                               c_dim.data()));

        size_t wsSize;
        cnnlGetFloorModWorkspaceSize(context->cnnlHandle(), aDesc, bDesc, cDesc,
                                     &wsSize);
        BangPtr wsData = context->getWorkspace(wsSize);

        cnnlStatus_t stat =
            cnnlFloorMod(context->cnnlHandle(), aDesc, aData, bDesc, bData,
                         cDesc, cData, wsData, wsSize);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(bDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

class SquaredDifferenceCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, bDesc, cDesc;
        auto a_dim = op->getInputs(0)->getDims();
        auto b_dim = op->getInputs(1)->getDims();
        auto c_dim = op->getOutput()->getDims();
        if (a_dim.size() == 0) {
            a_dim.push_back(1);
        }

        if (b_dim.size() == 0) {
            b_dim.push_back(1);
        }

        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, a_dim.size(),
                                               a_dim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(bDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, b_dim.size(),
                                               b_dim.data()));
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, c_dim.size(),
                                               c_dim.data()));

        size_t wsSize;
        cnnlGetSquaredDifferenceWorkspaceSize(context->cnnlHandle(), aDesc,
                                              bDesc, cDesc, &wsSize);
        BangPtr wsData = context->getWorkspace(wsSize);

        cnnlStatus_t stat =
            cnnlSquaredDifference(context->cnnlHandle(), aDesc, aData, bDesc,
                                  bData, cDesc, cData, wsData, wsSize);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(bDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

class AddCnnl : public ElementWiseCnnl {
    cnnlOpTensorDesc_t getOpType() const override { return CNNL_OP_TENSOR_ADD; }
};

class SubCnnl : public ElementWiseCnnl {
    cnnlOpTensorDesc_t getOpType() const override { return CNNL_OP_TENSOR_ADD; }
    tuple<float, float, float> getAlphBeta() const override {
        return {1.f, -1.f, 0.f};
    }
};

class MulCnnl : public ElementWiseCnnl {
    cnnlOpTensorDesc_t getOpType() const override { return CNNL_OP_TENSOR_MUL; }
};

class EqualCnnl : public LogicOpCnnl {
    cnnlLogicOp_t getOpType() const override { return CNNL_LOGIC_OP_EQ; }
};
class GreaterThanCnnl : public LogicOpCnnl {
    cnnlLogicOp_t getOpType() const override { return CNNL_LOGIC_OP_GT; }
};
class GreaterEqualCnnl : public LogicOpCnnl {
    cnnlLogicOp_t getOpType() const override { return CNNL_LOGIC_OP_GE; }
};
class LessThanCnnl : public LogicOpCnnl {
    cnnlLogicOp_t getOpType() const override { return CNNL_LOGIC_OP_LT; }
};
class LessEqualCnnl : public LogicOpCnnl {
    cnnlLogicOp_t getOpType() const override { return CNNL_LOGIC_OP_LE; }
};
class AndCnnl : public LogicOpCnnl {
    cnnlLogicOp_t getOpType() const override { return CNNL_LOGIC_OP_AND; }
};
class OrCnnl : public LogicOpCnnl {
    cnnlLogicOp_t getOpType() const override { return CNNL_LOGIC_OP_OR; }
};
class XorCnnl : public LogicOpCnnl {
    cnnlLogicOp_t getOpType() const override { return CNNL_LOGIC_OP_XOR; }
};
class NotCnnl : public LogicOpCnnl {
    cnnlLogicOp_t getOpType() const override { return CNNL_LOGIC_OP_NOT; }
};

class BitAndCnnl : public BitComputeCnnl {
    cnnlBitComputeOp_t getOpType() const override { return CNNL_CYCLE_BAND_OP; }
};
class BitOrCnnl : public BitComputeCnnl {
    cnnlBitComputeOp_t getOpType() const override { return CNNL_CYCLE_BOR_OP; }
};
class BitXorCnnl : public BitComputeCnnl {
    cnnlBitComputeOp_t getOpType() const override { return CNNL_CYCLE_BXOR_OP; }
};
class BitNotCnnl : public BitComputeCnnl {
    cnnlBitComputeOp_t getOpType() const override { return CNNL_BNOT_OP; }
};
// class BitLeftShiftCnnl : public BitComputeCnnl {
//     cnnlBitComputeOp_t getOpType() const override { return
//     CNNL_BLEFT_SHIFT_OP_V2; }
// };
// class BitRightShiftCnnl : public BitComputeCnnl {
//     cnnlBitComputeOp_t getOpType() const override { return
//     CNNL_BLEFT_SHIFT_OP_V2; }
// };

REGISTER_KERNEL(Device::BANG, OpType::Add, DataType::Float32, AddCnnl,
                "Add_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::Sub, DataType::Float32, SubCnnl,
                "Sub_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::Mul, DataType::Float32, MulCnnl,
                "Mul_cnnl_BANG_Float32");

REGISTER_KERNEL(Device::BANG, OpType::Div, DataType::Float32, DivCnnl,
                "Div_cnnl_Float32");
REGISTER_KERNEL(Device::BANG, OpType::Max, DataType::Float32, MaximumCnnl,
                "Maximum_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::Min, DataType::Float32, MinimumCnnl,
                "Minimum_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::MSELoss, DataType::Float32, MSELossCnnl,
                "MSELoss_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::Pow, DataType::Float32, PowerCnnl,
                "Power_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::FloorDiv, DataType::Float32, FloorDivCnnl,
                "FloorDiv_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::FloorMod, DataType::Float32, FloorModCnnl,
                "FloorMod_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::SquaredDifference, DataType::Float32,
                SquaredDifferenceCnnl, "SquaredDifference_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::Equal, DataType::Float32, EqualCnnl,
                "Equal_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::Greater, DataType::Float32,
                GreaterThanCnnl, "GreaterThan_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::GreaterOrEqual, DataType::Float32,
                GreaterEqualCnnl, "GreaterEqual_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::Less, DataType::Float32, LessThanCnnl,
                "LessThan_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::LessOrEqual, DataType::Float32,
                LessEqualCnnl, "LessEqual_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::And, DataType::Float32, AndCnnl,
                "And_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::Or, DataType::Float32, OrCnnl,
                "Or_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::Xor, DataType::Float32, XorCnnl,
                "Xor_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::Not, DataType::Float32, NotCnnl,
                "Not_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::BitwiseAnd, DataType::Float32, BitAndCnnl,
                "BitAnd_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::BitwiseOr, DataType::Float32, BitOrCnnl,
                "BitOr_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::BitwiseXor, DataType::Float32, BitXorCnnl,
                "BitXor_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::BitwiseNot, DataType::Float32, BitNotCnnl,
                "BitNot_cnnl_BANG_Float32");
// REGISTER_KERNEL(Device::BANG, OpType::BitLeftShift, DataType::Float32,
// BitLeftShiftCnnl,
//                 "BitLeftShift_cnnl_BANG_Float32");
// REGISTER_KERNEL(Device::BANG, OpType::BitRightShift, DataType::Float32,
// BitRightShiftCnnl,
//                 "BitRightShift_cnnl_BANG_Float32");
// REGISTER_KERNEL(Device::BANG, OpType::Pow, DataType::Float32,
// ElementWiseBang,
//                 "Pow_Bang_Float32");
}; // namespace infini
