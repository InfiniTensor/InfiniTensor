// #include "operators/convswish.h"
// #include "cuda/cuda_kernel_wihtout_config.h"
// #include "cuda/cuda_runtime.h"
// #include "cuda/cuda_utility.h"
// #include <tuple>

// namespace infini {
// class ConvSwishCudnn : public CudaKernelWithoutConfig {
//     std::tuple<void *, void *, void *, void *, cudnnTensorDescriptor_t,
//                cudnnFilterDescriptor_t, cudnnTensorDescriptor_t,
//                cudnnConvolutionDescriptor_t, cudnnActivationDescriptor_t,
//                cudnnTensorDescriptor_t>
//     createCuDNNDescriptor(const Ref<ConvSwishObj> &op) const {
//         void *const inData = (op->getInputs(0)->getRawDataPtr<void *>());
//         void *const knData = (op->getInputs(1)->getRawDataPtr<void *>());
//         void *const bData = (op->getInputs().size() > 2)
//                                 ? op->getInputs(2)->getRawDataPtr<void *>()
//                                 : nullptr;
//         auto cudnnDataType = cudnnDataTypeConvert(op->getDType());
//         void *const outData = (op->getOutput()->getRawDataPtr<void *>());

//         const auto [n, c, h, w, f, r, s] = op->getNCHWFRS();
//         const auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();

//         // get inputs
//         cudnnTensorDescriptor_t inDesc;
//         checkCudnnError(cudnnCreateTensorDescriptor(&inDesc));
//         checkCudnnError(cudnnSetTensor4dDescriptor(inDesc, CUDNN_TENSOR_NCHW,
//                                                    cudnnDataType, n, c, h,
//                                                    w));

//         // get kernels
//         cudnnFilterDescriptor_t knDesc;
//         checkCudnnError(cudnnCreateFilterDescriptor(&knDesc));
//         checkCudnnError(cudnnSetFilter4dDescriptor(
//             knDesc, cudnnDataType, CUDNN_TENSOR_NCHW, f, c, r, s));
//         // get bias
//         cudnnTensorDescriptor_t biasDesc;
//         checkCudnnError(cudnnCreateTensorDescriptor(&biasDesc));
//         checkCudnnError(cudnnSetTensor4dDescriptor(biasDesc,
//         CUDNN_TENSOR_NCHW,
//                                                    cudnnDataType, 1, f, 1,
//                                                    1));

//         // get convolution descriptor
//         cudnnConvolutionDescriptor_t convDesc;
//         checkCudnnError(cudnnCreateConvolutionDescriptor(&convDesc));
//         // TODO: CUDNN_CONVOLUTION is a tunable argument
//         checkCudnnError(cudnnSetConvolution2dDescriptor(
//             convDesc, ph, pw, sh, sw, dh, dw, CUDNN_CROSS_CORRELATION,
//             cudnnDataType));

//         // get activation descriptor
//         cudnnActivationDescriptor_t actDesc;
//         checkCudnnError(cudnnCreateActivationDescriptor(&actDesc));
//         // NOT_PROPAGATE_NAN is requierd by
//         // cudnnConvolotionBiasActivationForward
//         checkCudnnError(cudnnSetActivationDescriptor(
//             actDesc, CUDNN_ACTIVATION_SWISH, CUDNN_NOT_PROPAGATE_NAN, 0));

//         // get output descriptor
//         int outn, outc, outh, outw;
//         checkCudnnError(cudnnGetConvolution2dForwardOutputDim(
//             convDesc, inDesc, knDesc, &outn, &outc, &outh, &outw));
//         cudnnTensorDescriptor_t outDesc;
//         checkCudnnError(cudnnCreateTensorDescriptor(&outDesc));
//         checkCudnnError(cudnnSetTensor4dDescriptor(
//             outDesc, CUDNN_TENSOR_NCHW, cudnnDataType, outn, outc, outh,
//             outw));
//         IT_ASSERT((vector{outn, outc, outh, outw}) ==
//                       op->getOutput()->getDims(),
//                   "cuDNN output shape mismatches with OP output shape");

//         return tuple(inData, knData, bData, outData, inDesc, knDesc,
//         biasDesc,
//                      convDesc, actDesc, outDesc);
//     }
//     void compute(const Operator &_op,
//                  const RuntimeObj *_context) const override {
//         auto op = as<ConvSwishObj>(_op);
//         auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
//         const auto &[inData, knData, bData, outData, inDesc, knDesc,
//         biasDesc,
//                      convDesc, actDesc, outDesc] = createCuDNNDescriptor(op);
//         CudaPtr wsData = context->getWorkspace(100000);
//         float alpha1 = 1.f, alpha2 = 0.f;
//         checkCudnnError(cudnnConvolutionBiasActivationForward(
//             context->cudnnHandle(), &alpha1, inDesc, inData, knDesc, knData,
//             convDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, wsData,
//             100000, &alpha2, outDesc, outData, biasDesc, bData, actDesc,
//             outDesc, outData));
//         checkCudnnError(cudnnDestroyTensorDescriptor(outDesc));
//         checkCudnnError(cudnnDestroyActivationDescriptor(actDesc));
//         checkCudnnError(cudnnDestroyConvolutionDescriptor(convDesc));
//         checkCudnnError(cudnnDestroyTensorDescriptor(biasDesc));
//         checkCudnnError(cudnnDestroyFilterDescriptor(knDesc));
//         checkCudnnError(cudnnDestroyTensorDescriptor(inDesc));
//     }
// };

// REGISTER_KERNEL(Device::CUDA, OpType::ConvSwish, ConvSwishCudnn,
//                 "ConvSwish_cuDNN_CUDA");
// }; // namespace infini

#include "operators/convswish.h"
#include "conv2d_sigmoid_mul/conv2d_sigmoid_mul.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"

namespace infini {
class ConvSwishKernel : public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ConvSwishObj>(_op);
        const auto [n, c, h, w, f, r, s] = op->getNCHWFRS();
        const auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
        NineToothedTensor input = op->getInputs(0)->ToNineToothedTensor();
        NineToothedTensor weight = op->getInputs(1)->ToNineToothedTensor();
        NineToothedTensor bias = NineToothedTensor{};
        if (op->numInputs() > 2) {
            op->getInputs(2)->setShape(op->getOutput()->getDims());
            bias = op->getInputs(2)->ToNineToothedTensor({0, 1, 0, 0});
        }
        NineToothedTensor output = op->getOutput()->ToNineToothedTensor();
        launch_conv2d_sigmoid_mul(
            CUDAStream::getCurrentStream(), input, weight, bias, output,
            NineToothedTensor{}, NineToothedTensor{}, n, c, h, w, f, r, s, sh,
            sw, ph, pw, dh, dw, NineToothedDataType::NINETOOTHED_FLOAT32);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::ConvSwish, ConvSwishKernel,
                "ConvSwish_Nineth_CUDA");
} // namespace infini
