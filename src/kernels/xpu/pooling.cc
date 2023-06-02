#include "operators/pooling.h"
#include "xpu/xpu_kernel_without_config.h"
#include "xpu/xpu_runtime.h"

namespace infini {
class AvgPooling : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<PoolingObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto [n, c, h, w, kh, kw] = op->getNCHWRS();
        auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();

	std::vector<int> ksize = {kh, kw};
	std::vector<int> stride = {sh, sw};
	std::vector<int> pad = {ph, pw};

	auto ret = baidu::xpu::api::avg_pool2d<float>(context->XPUHandle(), (float*)aData, (float*)cData,
			                              n,c,h,w,ksize,stride,pad,true,true,nullptr,nullptr);
        assert(ret == 0);
        return;

    }
};

// class MaxPooling : public XPUKernelWithoutConfig {
//     void compute(const Operator &_op,
//                  const RuntimeObj *_context) const override {
//         auto op = as<PoolingObj>(_op);
//         auto context = dynamic_cast<const XPURuntimeObj *>(_context);
//         void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
//         void *const cData = (op->getOutput()->getRawDataPtr<void *>());
// 
//         auto [n, c, h, w, kh, kw] = op->getNCHWRS();
//         auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
// 
// 	std::vector<int> ksize = {kh, kw};
// 	std::vector<int> stride = {sh, sw};
// 	std::vector<int> pad = {ph, pw};
// 	int indices;
// 
// 	auto ret = baidu::xpu::api::max_pool2d<float>(context->XPUHandle(), (float*)aData, (float*)cData,
// 			                              &indices, n,c,h,w,ksize,stride,pad,true,nullptr,nullptr);
//         assert(ret == 0);
//         return;
// 
//     }
// };


// REGISTER_KERNEL(Device::XPU, OpType::MaxPool, DataType::Float32, MaxPooling,
//                 "MaxPool_xdnn_Float32");
REGISTER_KERNEL(Device::XPU, OpType::AvgPool, DataType::Float32, AvgPooling,
                "AvgPool_xdnn_Float32");
}; // namespace infini
