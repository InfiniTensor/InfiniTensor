#include "operators/topk.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"

namespace infini {
class TopKXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<TopKObj>(_op);
        auto input = op->getInputs(0);
        IT_ASSERT(input->getRank() == 2, "Kunlun Topk do not support dim != 2");

        auto output_0 = op->getOutput(0);
        auto output_1 = op->getOutput(1);
        void *const source = input->getRawDataPtr<void *>();
        void *const Values = output_0->getRawDataPtr<void *>();
        void *const Indices = output_1->getRawDataPtr<void *>();
        int Largest = op->getLargest();

        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);
        auto K = op->getTopk();

        int k = K[0];
        int m = input->getDims()[0];
        int n = input->getDims()[1];

        // Use workspace to save kernel output
        KUNLUNPtr workspace =
            context->getWorkspace(output_1->size() * sizeof(int));
        if (op->getOpType() == OpType::TopK) {
            if (op->getDType() == DataType::Float32) {
                checkKUNLUNError(xdnn::sorted_topk<float>(
                    context->KUNLUNHandle(), (float *)source, (float *)Values,
                    (int *)workspace, m, n, k, Largest));
            } else if (op->getDType() == DataType::Float16) {
                checkKUNLUNError(xdnn::sorted_topk<float16>(
                    context->KUNLUNHandle(), (float16 *)source,
                    (float16 *)Values, (int *)workspace, m, n, k, Largest));
            }
            // Cast to int64
            checkKUNLUNError((xdnn::cast<int, int64_t>(
                context->KUNLUNHandle(), (int *)workspace, (int64_t *)Indices,
                output_1->size())));
        } else {
            IT_TODO_HALT();
        }
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::TopK, TopKXdnn, "TopK_Xdnn");
}; // namespace infini
