#include "operators/element_wise.h"
#include "bang/bang_element_wise.h"
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
        auto dim = op->getInputs(0)->getDims();
        if (dim.size() != 4)
            IT_TODO_HALT();

        int dim_array[4] = {dim[0], dim[1], dim[2], dim[3]};
        // get inputs
        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(bDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        // get outputs
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        // get op descriptor
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

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(bDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
        checkCnnlError(cnnlDestroyOpTensorDescriptor(opDesc));
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
        auto dim = op->getInputs(0)->getDims();
        if (dim.size() != 4)
            IT_TODO_HALT();

        int dim_array[4] = {dim[0], dim[1], dim[2], dim[3]};
        // get inputs
        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(bDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        // get outputs
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        size_t wsSize;
        cnnlGetDivWorkspaceSize(context->cnnlHandle(), aDesc, bDesc, cDesc,
                                     &wsSize);

        BangPtr wsData = context->getWorkspace(wsSize);

        cnnlStatus_t stat = cnnlDiv_v2(context->cnnlHandle(),
                                       CNNL_COMPUTATION_HIGH_PRECISION,
                                       aDesc, aData, bDesc, bData, wsData, wsSize, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(bDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

class DivNoNanCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, bDesc, cDesc;
        auto dim = op->getInputs(0)->getDims();
        if (dim.size() != 4)
            IT_TODO_HALT();

        int dim_array[4] = {dim[0], dim[1], dim[2], dim[3]};
        // get inputs
        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(bDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        // get outputs
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        size_t wsSize;
        cnnlGetDivNoNanWorkspaceSize(context->cnnlHandle(), aDesc, bDesc, cDesc,
                                     &wsSize);

        BangPtr wsData = context->getWorkspace(wsSize);

        cnnlStatus_t stat = cnnlDivNoNan_v2(context->cnnlHandle(),
                                       CNNL_COMPUTATION_HIGH_PRECISION,
                                       aDesc, aData, bDesc, bData, wsData, wsSize, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
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
        auto dim = op->getInputs(0)->getDims();
        if (dim.size() != 4)
            IT_TODO_HALT();

        int dim_array[4] = {dim[0], dim[1], dim[2], dim[3]};
        // get inputs
        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(bDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        // get outputs
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        // get op descriptor
        size_t wsSize;
        cnnlGetMaximumWorkspaceSize(context->cnnlHandle(), cDesc, &wsSize);
        BangPtr wsData = context->getWorkspace(wsSize);

        cnnlStatus_t stat = cnnlMaximum(context->cnnlHandle(), aDesc, aData, bDesc, bData,
                                        cDesc, cData, wsData, wsSize);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
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
        auto dim = op->getInputs(0)->getDims();
        if (dim.size() != 4)
            IT_TODO_HALT();

        int dim_array[4] = {dim[0], dim[1], dim[2], dim[3]};
        // get inputs
        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(bDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        // get outputs
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        // get op descriptor
        size_t wsSize;
        cnnlGetMinimumWorkspaceSize(context->cnnlHandle(), cDesc, &wsSize);
        BangPtr wsData = context->getWorkspace(wsSize);

        cnnlStatus_t stat = cnnlMinimum(context->cnnlHandle(), aDesc, aData, bDesc, bData,
                                        cDesc, cData, wsData, wsSize);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
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
        auto dim = op->getInputs(0)->getDims();
        if (dim.size() != 4)
            IT_TODO_HALT();

        int dim_array[4] = {dim[0], dim[1], dim[2], dim[3]};
        int dim_out[4] ={1,1,1,1};
        // get inputs
        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(bDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        // get outputs
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        if ( reduction == MSELossObj::None ) {
          checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                                 CNNL_DTYPE_FLOAT, 4, dim_array));
        } else {
          checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                                 CNNL_DTYPE_FLOAT, 4, dim_out));
        }
        cnnlStatus_t stat;
        if( reduction == MSELossObj::None ) {
          stat = cnnlMSELoss(context->cnnlHandle(), CNNL_MSE_LOSS_NONE, aDesc, aData, bDesc, bData,
                             cDesc, cData);
        } else if (reduction == MSELossObj::Sum) {
          stat = cnnlMSELoss(context->cnnlHandle(), CNNL_MSE_LOSS_SUM, aDesc, aData, bDesc, bData,
                             cDesc, cData);
        } else {
          stat = cnnlMSELoss(context->cnnlHandle(), CNNL_MSE_LOSS_MEAN, aDesc, aData, bDesc, bData,
                             cDesc, cData);
        }

        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
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
        auto dim = op->getInputs(0)->getDims();
        if (dim.size() != 4)
            IT_TODO_HALT();

        int dim_array[4] = {dim[0], dim[1], dim[2], dim[3]};
        // get inputs
        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(bDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        // get outputs
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        // get op descriptor
        size_t wsSize;
        cnnlGetPowWorkspaceSize(context->cnnlHandle(), aDesc, bDesc, cDesc, &wsSize);
        BangPtr wsData = context->getWorkspace(wsSize);

        cnnlStatus_t stat = cnnlPow(context->cnnlHandle(), CNNL_COMPUTATION_HIGH_PRECISION,
                                    aDesc, aData, bDesc, bData, wsData, wsSize, cDesc, cData);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
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
        auto dim = op->getInputs(0)->getDims();
        if (dim.size() != 4)
            IT_TODO_HALT();

        int dim_array[4] = {dim[0], dim[1], dim[2], dim[3]};
        // get inputs
        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(bDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        // get outputs
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        size_t wsSize;
        cnnlGetFloorDivWorkspaceSize(context->cnnlHandle(), aDesc, bDesc, cDesc,
                                     &wsSize);

        BangPtr wsData = context->getWorkspace(wsSize);

        cnnlStatus_t stat = cnnlFloorDiv_v2(context->cnnlHandle(),
                                       CNNL_COMPUTATION_HIGH_PRECISION,
                                       aDesc, aData, bDesc, bData, cDesc, cData, wsData, wsSize);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(bDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

class FloorDivTruncCnnl : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ElementWiseObj>(_op);
        auto context = dynamic_cast<const BangRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        cnnlTensorDescriptor_t aDesc, bDesc, cDesc;
        auto dim = op->getInputs(0)->getDims();
        if (dim.size() != 4)
            IT_TODO_HALT();

        int dim_array[4] = {dim[0], dim[1], dim[2], dim[3]};
        // get inputs
        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(bDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        // get outputs
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        size_t wsSize;
        cnnlGetFloorDivTruncWorkspaceSize(context->cnnlHandle(), aDesc, bDesc, cDesc,
                                     &wsSize);

        BangPtr wsData = context->getWorkspace(wsSize);

        cnnlStatus_t stat = cnnlFloorDivTrunc(context->cnnlHandle(),
                                       CNNL_COMPUTATION_HIGH_PRECISION,
                                       aDesc, aData, bDesc, bData, cDesc, cData, wsData, wsSize);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
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
        auto dim = op->getInputs(0)->getDims();
        if (dim.size() != 4)
            IT_TODO_HALT();

        int dim_array[4] = {dim[0], dim[1], dim[2], dim[3]};
        // get inputs
        checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
        checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
        checkCnnlError(cnnlSetTensorDescriptor(bDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        // get outputs
        checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
        checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
                                               CNNL_DTYPE_FLOAT, 4, dim_array));

        size_t wsSize;
        cnnlGetFloorModWorkspaceSize(context->cnnlHandle(), aDesc, bDesc, cDesc,
                                     &wsSize);

        BangPtr wsData = context->getWorkspace(wsSize);

        cnnlStatus_t stat = cnnlFloorMod(context->cnnlHandle(),
                                       aDesc, aData, bDesc, bData, cDesc, cData, wsData, wsSize);
        if (stat != CNNL_STATUS_SUCCESS)
            return;

        // Destories in BANG does not require sync. But cnnl does not state
        // whether sync is required before destories.
        checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(bDesc));
        checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
    }
};

// class FloorModTruncCnnl : public BangKernelWithoutConfig {
//     void compute(const Operator &_op,
//                  const RuntimeObj *_context) const override {
//         auto op = as<ElementWiseObj>(_op);
//         auto context = dynamic_cast<const BangRuntimeObj *>(_context);
// 
//         void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
//         void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
//         void *const cData = (op->getOutput()->getRawDataPtr<void *>());
// 
//         cnnlTensorDescriptor_t aDesc, bDesc, cDesc;
//         auto dim = op->getInputs(0)->getDims();
//         if (dim.size() != 4)
//             IT_TODO_HALT();
// 
//         int dim_array[4] = {dim[0], dim[1], dim[2], dim[3]};
//         // get inputs
//         checkCnnlError(cnnlCreateTensorDescriptor(&aDesc));
//         checkCnnlError(cnnlSetTensorDescriptor(aDesc, CNNL_LAYOUT_NCHW,
//                                                CNNL_DTYPE_FLOAT, 4, dim_array));
// 
//         checkCnnlError(cnnlCreateTensorDescriptor(&bDesc));
//         checkCnnlError(cnnlSetTensorDescriptor(bDesc, CNNL_LAYOUT_NCHW,
//                                                CNNL_DTYPE_FLOAT, 4, dim_array));
// 
//         // get outputs
//         checkCnnlError(cnnlCreateTensorDescriptor(&cDesc));
//         checkCnnlError(cnnlSetTensorDescriptor(cDesc, CNNL_LAYOUT_NCHW,
//                                                CNNL_DTYPE_FLOAT, 4, dim_array));
// 
//         size_t wsSize;
//         cnnlGetFloorModTruncWorkspaceSize(context->cnnlHandle(), aDesc, bDesc, cDesc,
//                                      &wsSize);
// 
//         BangPtr wsData = context->getWorkspace(wsSize);
// 
//         cnnlStatus_t stat = cnnlFloorModTrunc(context->cnnlHandle(),
//                                        aDesc, aData, bDesc, bData, cDesc, cData, wsData, wsSize);
//         if (stat != CNNL_STATUS_SUCCESS)
//             return;
// 
//         // Destories in BANG does not require sync. But cnnl does not state
//         // whether sync is required before destories.
//         checkCnnlError(cnnlDestroyTensorDescriptor(aDesc));
//         checkCnnlError(cnnlDestroyTensorDescriptor(bDesc));
//         checkCnnlError(cnnlDestroyTensorDescriptor(cDesc));
//     }
// };

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

class ElementWiseBang : public BangKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        element_wise_kernel(_context, _op);
    }
};

REGISTER_KERNEL(Device::BANG, OpType::Add, DataType::Float32, AddCnnl,
                "Add_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::Sub, DataType::Float32, SubCnnl,
                "Sub_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::Mul, DataType::Float32, MulCnnl,
                "Mul_cnnl_BANG_Float32");

REGISTER_KERNEL(Device::BANG, OpType::DivDemo, DataType::Float32, ElementWiseBang,
                "DivDemo_Bang_Float32");
REGISTER_KERNEL(Device::BANG, OpType::Div, DataType::Float32, DivCnnl,
                "Div_cnnl_Float32");
REGISTER_KERNEL(Device::BANG, OpType::DivNoNan, DataType::Float32, DivNoNanCnnl,
                "DivNoNan_cnnl_Float32");
REGISTER_KERNEL(Device::BANG, OpType::Maximum, DataType::Float32, MaximumCnnl,
                "Maximum_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::Minimum, DataType::Float32, MinimumCnnl,
                "Minimum_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::MSELoss, DataType::Float32, MSELossCnnl,
                "MSELoss_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::Power, DataType::Float32, PowerCnnl,
                "Power_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::FloorDiv, DataType::Float32, FloorDivCnnl,
                "FloorDiv_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::FloorDivTrunc, DataType::Float32, FloorDivTruncCnnl,
                "FloorDivTrunc_cnnl_BANG_Float32");
REGISTER_KERNEL(Device::BANG, OpType::FloorMod, DataType::Float32, FloorModCnnl,
                "FloorMod_cnnl_BANG_Float32");
// REGISTER_KERNEL(Device::BANG, OpType::FloorModTrunc, DataType::Float32, FloorModTruncCnnl,
//                 "FloorModTrunc_cnnl_BANG_Float32");
// REGISTER_KERNEL(Device::BANG, OpType::Pow, DataType::Float32,
// ElementWiseBang,
//                 "Pow_Bang_Float32");
}; // namespace infini
