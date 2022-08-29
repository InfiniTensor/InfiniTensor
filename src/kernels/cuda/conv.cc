#include "operators/conv.h"
#include "core/kernel.h"
#include "cuda/cuda_runtime.h"

namespace infini {

static constexpr int N_ALGO = 8;
static constexpr cudnnConvolutionFwdAlgo_t ALGOS[N_ALGO] = {
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED};

struct ConvCuDnnPerfRecord : public PerfRecord {
    int algo = 0; // cudnnConvolutionFwdAlgo_t
    size_t workspaceSize = 100000;
    bool fuseAct = false;
};

class convCudnn : public Kernel {

    bool cuDNNUnfused(const Ref<ConvObj> &op, const ConvCuDnnPerfRecord &record,
                      const CudaRuntimeObj *context) const {
        cudnnStatus_t stat;
        void *const inData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const knData = (op->getInputs(1)->getRawDataPtr<void *>());
        if (op->getInputs().size() > 2) // Bias is not supported yet
            IT_TODO_HALT();
        // void *const biasData = (op->getInputs(2)->getRawDataPtr<void *>());
        void *const outData = (op->getOutput()->getRawDataPtr<void *>());

        const auto [n, c, h, w, f, r, s] = op->getNCHWFRS();
        const int cpg = op->getChannelPerGroup();
        const int g = c / cpg;
        const auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();

        int channelsPerGrp = cpg, channels = c;

        // get inputs
        cudnnTensorDescriptor_t inDesc;
        checkCudnnError(cudnnCreateTensorDescriptor(&inDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(
            inDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, channels, h, w));

        // get kernels
        cudnnFilterDescriptor_t knDesc;
        checkCudnnError(cudnnCreateFilterDescriptor(&knDesc));
        checkCudnnError(cudnnSetFilter4dDescriptor(knDesc, CUDNN_DATA_FLOAT,
                                                   CUDNN_TENSOR_NCHW, f,
                                                   channelsPerGrp, r, s));

        // get bias
        cudnnTensorDescriptor_t biasDesc;
        checkCudnnError(cudnnCreateTensorDescriptor(&biasDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(
            biasDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, f, 1, 1));

        // get convlution descriptor
        cudnnConvolutionDescriptor_t convDesc;
        checkCudnnError(cudnnCreateConvolutionDescriptor(&convDesc));
        // TODO: CUDNN_CONVOLUTION is a tunable argument
        checkCudnnError(cudnnSetConvolution2dDescriptor(
            convDesc, ph, pw, sh, sw, dh, dw, CUDNN_CROSS_CORRELATION,
            CUDNN_DATA_FLOAT));
        if (g > 1) {
            checkCudnnError(cudnnSetConvolutionGroupCount(convDesc, g));
        }

        // get activation descriptor
        cudnnActivationDescriptor_t actDesc;
        checkCudnnError(cudnnCreateActivationDescriptor(&actDesc));
        // NOT_PROPAGATE_NAN is requierd by
        // cudnnConvolotionBiasActivationForward
        switch (op->getAct()) {
        case ActType::Relu:
            checkCudnnError(cudnnSetActivationDescriptor(
                actDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0));
            break;
        case ActType::Sigmoid:
            checkCudnnError(cudnnSetActivationDescriptor(
                actDesc, CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0));
            break;
        case ActType::None:
            checkCudnnError(
                cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_IDENTITY,
                                             CUDNN_NOT_PROPAGATE_NAN, 0));
            break;
        default:
            assert(false);
        }

        // get outputs
        int outn, outc, outh, outw;
        checkCudnnError(cudnnGetConvolution2dForwardOutputDim(
            convDesc, inDesc, knDesc, &outn, &outc, &outh, &outw));
        cudnnTensorDescriptor_t outDesc;
        checkCudnnError(cudnnCreateTensorDescriptor(&outDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(outDesc, CUDNN_TENSOR_NCHW,
                                                   CUDNN_DATA_FLOAT, outn, outc,
                                                   outh, outw));
        IT_ASSERT((vector{outn, outc, outh, outw}) ==
                      op->getOutput()->getDims(),
                  "cuDNN output shape mismatches with OP output shape");

        // get workspace
        size_t wsSize = record.workspaceSize;
        stat = cudnnGetConvolutionForwardWorkspaceSize(
            context->cudnnHandle(), inDesc, knDesc, convDesc, outDesc,
            ALGOS[record.algo], &wsSize);
        if (stat != CUDNN_STATUS_SUCCESS)
            return false;
        // assert(wsSize < (size_t)3 * 1024 * 1024 * 1024);
        // if (wsSize >= (size_t)10 * 1024 * 1024 * 1024)
        //     continue;
        CudaPtr wsData = context->getWorkspace(wsSize);
        float alpha = 1.f, beta = 0.f;

        stat = cudnnConvolutionForward(context->cudnnHandle(), &alpha, inDesc,
                                       inData, knDesc, knData, convDesc,
                                       ALGOS[record.algo], wsData, wsSize,
                                       &beta, outDesc, outData);
        if (stat != CUDNN_STATUS_SUCCESS)
            return false;
        // TODO:
        // // bias
        // if (bias != nullptr) {
        //     auto sz = op.getOutputs()[0]->size();
        //     // TODO: element wise
        //     t += sz * 2 / 400;
        // }
        // // act
        // if (act != None) {
        //     stat = cudnnActivationForward(cudnnHandle(), actDesc,
        //                                   &alpha, inDesc, inData,
        //                                   &beta, outDesc, outData);
        //     checkCudaError(cudaDeviceSynchronize());
        //     end = ch::high_resolution_clock::now();
        //     if (stat != CUDNN_STATUS_SUCCESS) {
        //         durtime = INFINITY;
        //         break;
        //     }
        //     t +=
        //         ch::duration_cast<ch::duration<double>>(end -
        //         beg).count() * 1000; // ms
        // }

        // best = ConvResult{durtime, ALGOS[i], wsSize, false};

        // // w/ bias & act
        // for (int j = 0; j < rounds + warmupRounds; ++j) {
        //     cudnnStatus_t stat;
        //     if (j == warmupRounds) {
        //         checkCudaError(cudaDeviceSynchronize());
        //         beg = ch::high_resolution_clock::now();
        //     }
        //     stat = cudnnConvolutionBiasActivationForward(
        //         cudnnHandle(), &alpha, inDesc, inData, knDesc, knData,
        //         convDesc, ALGOS[i], wsData, wsSize, &beta, outDesc,
        //         outData, biasDesc, biasData, actDesc, outDesc, outData);
        //     if (stat != CUDNN_STATUS_SUCCESS) {
        //         // checkCudnnError(stat);
        //         // Do not checkCudnnError since not all algorithms are
        //         // supported
        //         durtime_fuse = INFINITY;
        //         break;
        //     }
        // }

        // Destories in CUDA does not require sync. But cuDNN does not state
        // whether sync is required before destories.
        checkCudnnError(cudnnDestroyTensorDescriptor(inDesc));
        checkCudnnError(cudnnDestroyTensorDescriptor(outDesc));
        checkCudnnError(cudnnDestroyFilterDescriptor(knDesc));
        checkCudnnError(cudnnDestroyConvolutionDescriptor(convDesc));
        return true;
    }

    void compute(const Operator &op, const RuntimeObj *context) const override {
        ConvCuDnnPerfRecord record; // with paramters in default ctor
        compute(op, record, context);
    }

    PerfRecord tune(const Operator &_op,
                    const RuntimeObj *_context) const override {
        // TODO: real tuning
        ConvCuDnnPerfRecord ret;
        ret.time = timeit([&]() { compute(_op, _context); });
        return ret;
    }

    void compute(const Operator &_op, const PerfRecord &_record,
                 const RuntimeObj *_context) const override {
        auto op = as<ConvObj>(_op);
        auto &record = dynamic_cast<const ConvCuDnnPerfRecord &>(_record);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        bool success = cuDNNUnfused(op, record, context);
        IT_ASSERT(success);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Conv, DataType::Float32, convCudnn,
                "Conv_cuDNN_CUDA_Float32");

} // namespace infini