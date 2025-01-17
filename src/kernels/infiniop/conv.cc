// #include "operators/conv.h"
// #include "core/kernel.h"

// namespace infini {

// class ConvOp : public Kernel {
//     void compute(const Operator &_op,
//                  const RuntimeObj *context) const override {
//         auto op = as<ConvObj>(_op);
//         void *const xData = (op->getInputs(0)->getRawDataPtr<void *>());
//         void *const wData = (op->getInputs(1)->getRawDataPtr<void *>());
//         void *const yData = (op->getOutput()->getRawDataPtr<void *>());
//         uint64_t workspace_size = 0;
//         if (op->numInputs() == 2) {
//             CHECK_ERROR(infiniopGetConvWorkspaceSize(
//                 (infiniopConvDescriptor_t)op->getOpDesc(), &workspace_size));
//             IT_ASSERT(workspace_size <= context->getWorkspaceSize());
//             void *workspace = context->getWorkspace(workspace_size);
//             CHECK_ERROR(infiniopConv((infiniopConvDescriptor_t)op->getOpDesc(),
//                                      workspace, workspace_size, yData, xData,
//                                      wData, context->getCurrentStream()));
//         } else if (op->numInputs() == 3) {
//             void *const bData = (op->getInputs(2)->getRawDataPtr<void *>());
//             CHECK_ERROR(infiniopGetConvBiasActWorkspaceSize(
//                 (infiniopConvBiasActDescriptor_t)op->getOpDesc(),
//                 &workspace_size));
//             IT_ASSERT(workspace_size <= context->getWorkspaceSize());
//             void *workspace = context->getWorkspace(workspace_size);
//             CHECK_ERROR(infiniopConvBiasAct(
//                 (infiniopConvBiasActDescriptor_t)op->getOpDesc(), workspace,
//                 workspace_size, yData, xData, wData, bData,
//                 context->getCurrentStream()));
//         } else {
//             IT_ASSERT(false);
//         }
//     }

//     PerfRecord tune(const Operator &_op,
//                     const RuntimeObj *context) const override {
//         // TODO: tune should be in infiniop
//         return PerfRecord();
//     }

//     void compute(const Operator &op, const PerfRecord &record,
//                  const RuntimeObj *context) const override {
//         compute(op, context);
//     }
// };

// REGISTER_KERNEL(Device::CUDA, OpType::Conv, ConvOp, "Conv_infiniop_cuda");
// REGISTER_KERNEL(Device::CPU, OpType::Conv, ConvOp, "Conv_infiniop_cpu");
// }; // namespace infini
