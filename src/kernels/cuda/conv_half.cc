#include "core/kernel.h"
#include "cuda/cuda_runtime.h"
#include "operators/conv.h"
#include <chrono>
#include <functional>
#include <limits>
#include <tuple>

namespace infini {

struct ConvCuDnnPerfRecordObj : public PerfRecordObj {
    int algo = 0; // cudnnConvolutionFwdAlgo_t
    int mode = 1;
    size_t workspaceSize = 100000;
    bool fuseAct = false;
    void to_json(json &j) override {
        j["type"] = 1;
        j["data"] = std::make_tuple(algo, mode, fuseAct, time, workspaceSize);
    }
    static PerfRecord from_json(const json &j) {
        ConvCuDnnPerfRecordObj tmp;
        auto [Algo, Mode, FuseAct, Time, WorkspaceSize] =
            j["data"].get<tuple<int, int, bool, double, size_t>>();
        tmp.algo = Algo;
        tmp.mode = Mode;
        tmp.fuseAct = FuseAct;
        tmp.time = Time;
        tmp.workspaceSize = WorkspaceSize;
        return make_ref<ConvCuDnnPerfRecordObj>(tmp);
    }
};

using ConvCuDnnPerfRecord = Ref<ConvCuDnnPerfRecordObj>;

class convCudnnFP16 : public Kernel {

    static constexpr int N_ALGO = 8;
    static constexpr int N_MODE = 2;
    static constexpr cudnnConvolutionFwdAlgo_t ALGOS[8] = {
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        CUDNN_CONVOLUTION_FWD_ALGO_FFT,
        CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED};

    static constexpr cudnnConvolutionMode_t MODES[2] = {
        CUDNN_CONVOLUTION, CUDNN_CROSS_CORRELATION};

    std::tuple<void *, void *, void *, cudnnTensorDescriptor_t,
               cudnnFilterDescriptor_t, cudnnTensorDescriptor_t,
               cudnnConvolutionDescriptor_t, cudnnActivationDescriptor_t,
               cudnnTensorDescriptor_t>
    createCuDNNDescriptor(const Ref<ConvObj> &op,
                          const ConvCuDnnPerfRecord &record) const {
        void *const inData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const knData = (op->getInputs(1)->getRawDataPtr<void *>());
        // Bias is not supported yet
        if (op->getInputs().size() > 2) {
            IT_TODO_HALT();
        }
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
        checkCudnnError(cudnnSetTensor4dDescriptor(inDesc, CUDNN_TENSOR_NCHW,
                                                   CUDNN_DATA_HALF, n, channels,
                                                   h, w)); /*fp16 type*/

        // get kernels
        cudnnFilterDescriptor_t knDesc;
        checkCudnnError(cudnnCreateFilterDescriptor(&knDesc));
        checkCudnnError(cudnnSetFilter4dDescriptor(
            knDesc, CUDNN_DATA_HALF, /*fp16 type*/
            CUDNN_TENSOR_NCHW, f, channelsPerGrp, r, s));
        // get bias
        cudnnTensorDescriptor_t biasDesc;
        checkCudnnError(cudnnCreateTensorDescriptor(&biasDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW,
                                                   CUDNN_DATA_HALF, 1, f, 1,
                                                   1)); /*fp16 type*/

        // get convolution descriptor
        cudnnConvolutionDescriptor_t convDesc;
        checkCudnnError(cudnnCreateConvolutionDescriptor(&convDesc));
        // TODO: CUDNN_CONVOLUTION is a tunable argument
        checkCudnnError(cudnnSetConvolution2dDescriptor(
            convDesc, ph, pw, sh, sw, dh, dw, MODES[record->mode],
            CUDNN_DATA_HALF)); /*fp16 type*/
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

        // get output descriptor
        int outn, outc, outh, outw;
        checkCudnnError(cudnnGetConvolution2dForwardOutputDim(
            convDesc, inDesc, knDesc, &outn, &outc, &outh, &outw));
        cudnnTensorDescriptor_t outDesc;
        checkCudnnError(cudnnCreateTensorDescriptor(&outDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(outDesc, CUDNN_TENSOR_NCHW,
                                                   CUDNN_DATA_HALF, outn, outc,
                                                   outh, outw));
        IT_ASSERT((vector{outn, outc, outh, outw}) ==
                      op->getOutput()->getDims(),
                  "cuDNN output shape mismatches with OP output shape");

        return tuple(inData, knData, outData, inDesc, knDesc, biasDesc,
                     convDesc, actDesc, outDesc);
    }

    bool cuDNNUnfused(const Ref<ConvObj> &op, const ConvCuDnnPerfRecord &record,
                      const CudaRuntimeObj *context) const {
        cudnnStatus_t stat;

        const auto &[inData, knData, outData, inDesc, knDesc, biasDesc,
                     convDesc, actDesc, outDesc] =
            createCuDNNDescriptor(op, record);
        size_t wsSize = record->workspaceSize;
        CudaPtr wsData = context->getWorkspace(wsSize);
        float alpha = 1.f, beta = 0.f;

        stat = cudnnConvolutionForward(context->cudnnHandle(), &alpha, inDesc,
                                       inData, knDesc, knData, convDesc,
                                       ALGOS[record->algo], wsData, wsSize,
                                       &beta, outDesc, outData);
        if (stat != CUDNN_STATUS_SUCCESS) {
            return false;
        }
        checkCudnnError(cudnnDestroyTensorDescriptor(outDesc));
        checkCudnnError(cudnnDestroyActivationDescriptor(actDesc));
        checkCudnnError(cudnnDestroyConvolutionDescriptor(convDesc));
        checkCudnnError(cudnnDestroyTensorDescriptor(biasDesc));
        checkCudnnError(cudnnDestroyFilterDescriptor(knDesc));
        checkCudnnError(cudnnDestroyTensorDescriptor(inDesc));
        return true;
    }

    void compute(const Operator &op, const RuntimeObj *context) const override {
        auto record = make_ref<ConvCuDnnPerfRecordObj>(); // with paramters in
                                                          // default ctor
        compute(op, record, context);
    }

    PerfRecord tune(const Operator &_op,
                    const RuntimeObj *_context) const override {
        ConvCuDnnPerfRecordObj ret;
        ret.time = std::numeric_limits<double>::max();
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        auto op = as<ConvObj>(_op);
        // Both modes have the same performance. Only run cross-correlation.
        for (int mode = 1; mode < 2; mode++) {
            // Try every possible algorithm of convolution
            for (int algo = 0; algo < N_ALGO; algo++) {
                auto recordRef = make_ref<ConvCuDnnPerfRecordObj>();
                auto &record = *recordRef;
                record.mode = mode;
                record.algo = algo;
                cudnnStatus_t stat;
                const auto &[inData, knData, outData, inDesc, knDesc, biasDesc,
                             convDesc, actDesc, outDesc] =
                    createCuDNNDescriptor(op, recordRef);

                // get workspace
                stat = cudnnGetConvolutionForwardWorkspaceSize(
                    context->cudnnHandle(), inDesc, knDesc, convDesc, outDesc,
                    ALGOS[record.algo], &record.workspaceSize);
                if (stat != CUDNN_STATUS_SUCCESS) {
                    continue;
                }
                if (record.workspaceSize > context->getWorkspaceSize()) {
                    continue;
                }
                CudaPtr wsData = context->getWorkspace(record.workspaceSize);
                float alpha = 1.f, beta = 0.f;

                stat = cudnnConvolutionForward(
                    context->cudnnHandle(), &alpha, inDesc, inData, knDesc,
                    knData, convDesc, ALGOS[record.algo], wsData,
                    record.workspaceSize, &beta, outDesc, outData);
                if (stat != CUDNN_STATUS_SUCCESS) {
                    continue;
                }
                record.time = timeit(
                    [&]() {
                        cudnnConvolutionForward(context->cudnnHandle(), &alpha,
                                                inDesc, inData, knDesc, knData,
                                                convDesc, ALGOS[record.algo],
                                                wsData, record.workspaceSize,
                                                &beta, outDesc, outData);
                    },
                    [&]() { context->sync(); });
                // printf("mode:%d algo:%d :%.8lf\n", mode, algo, record.time);

                // Update the tune result
                if (ret.time > record.time) {
                    ret = record;
                }
                checkCudnnError(cudnnDestroyTensorDescriptor(outDesc));
                checkCudnnError(cudnnDestroyActivationDescriptor(actDesc));
                checkCudnnError(cudnnDestroyConvolutionDescriptor(convDesc));
                checkCudnnError(cudnnDestroyTensorDescriptor(biasDesc));
                checkCudnnError(cudnnDestroyFilterDescriptor(knDesc));
                checkCudnnError(cudnnDestroyTensorDescriptor(inDesc));
            }
        }
        // printf("the best algo is %d, the best conv mode is %d\n", ret.algo,
        //        ret.mode);
        IT_ASSERT(ret.time < std::numeric_limits<double>::max(), "No valid "
                                                                 "algorithm "
                                                                 "found");
        return make_ref<ConvCuDnnPerfRecordObj>(ret);
    }

    void compute(const Operator &_op, const PerfRecord &_record,
                 const RuntimeObj *_context) const override {
        auto op = as<ConvObj>(_op);
        auto record = as<ConvCuDnnPerfRecordObj>(_record);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        bool success = cuDNNUnfused(op, record, context);
        IT_ASSERT(success);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Conv, DataType::Float16, convCudnnFP16,
                "Conv_cuDNN_CUDA_Float16");

} // namespace infini
