#include "core/kernel.h"
#include "operators/conv.h"

#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"

#include <chrono>
#include <functional>
#include <limits>
#include <tuple>

namespace infini {

struct Conv3dCuDnnPerfRecordObj : public PerfRecordObj {
    int algo = 0; // cudnnConvolutionFwdAlgo_t
    int mode = 1;
    size_t workspaceSize = 100000;
    bool fuseAct = false;
    void to_json(json &j) override {
        j["type"] = 1;
        j["data"] = std::make_tuple(algo, mode, fuseAct, time, workspaceSize);
    }
    static PerfRecord from_json(const json &j) {
        Conv3dCuDnnPerfRecordObj tmp;
        auto [Algo, Mode, FuseAct, Time, WorkspaceSize] =
            j["data"].get<tuple<int, int, bool, double, size_t>>();
        tmp.algo = Algo;
        tmp.mode = Mode;
        tmp.fuseAct = FuseAct;
        tmp.time = Time;
        tmp.workspaceSize = WorkspaceSize;
        return make_ref<Conv3dCuDnnPerfRecordObj>(tmp);
    }
};

using Conv3dCuDnnPerfRecord = Ref<Conv3dCuDnnPerfRecordObj>;

class Conv3dCudnn : public Kernel {
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
    createCuDNNDescriptor(const Ref<Conv3dObj> &op,
                          const Conv3dCuDnnPerfRecord &record) const {
        constexpr auto numDim = 5;
        void *const inData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const knData = (op->getInputs(1)->getRawDataPtr<void *>());
        // Bias is not supported yet
        if (op->getInputs().size() > 2) {
            IT_TODO_HALT();
        }
        auto cudnnDataType = cudnnDataTypeConvert(op->getDType());
        // void *const biasData = (op->getInputs(2)->getRawDataPtr<void *>());
        void *const outData = (op->getOutput()->getRawDataPtr<void *>());

        const auto [n, c, d, h, w, f, q, r, s] = op->getNCDHWFQRS();
        const int cpg = op->getChannelPerGroup();
        const int g = c / cpg;
        const auto [pd, ph, pw, sd, sh, sw, dd, dh, dw] =
            op->getPadStrideDilation();

        const int inDim[] = {n, c, d, h, w};
        const int knDim[] = {f, cpg, q, r, s};
        const int biasDim[] = {1, f, 1, 1, 1};
        int pad[] = {pd, ph, pw};
        int stride[] = {sd, sh, sw};
        int dilation[] = {dd, dh, dw};

        // get inputs
        cudnnTensorDescriptor_t inDesc;
        checkCudnnError(cudnnCreateTensorDescriptor(&inDesc));
        checkCudnnError(cudnnSetTensorNdDescriptorEx(
            inDesc, CUDNN_TENSOR_NCHW, cudnnDataType, numDim, inDim));

        // get kernels
        cudnnFilterDescriptor_t knDesc;
        checkCudnnError(cudnnCreateFilterDescriptor(&knDesc));
        checkCudnnError(cudnnSetFilterNdDescriptor(
            knDesc, cudnnDataType, CUDNN_TENSOR_NCHW, numDim, knDim));

        // get bias
        cudnnTensorDescriptor_t biasDesc;
        checkCudnnError(cudnnCreateTensorDescriptor(&biasDesc));
        checkCudnnError(cudnnSetTensorNdDescriptorEx(
            biasDesc, CUDNN_TENSOR_NCHW, cudnnDataType, numDim, biasDim));

        // get convolution descriptor
        cudnnConvolutionDescriptor_t convDesc;
        checkCudnnError(cudnnCreateConvolutionDescriptor(&convDesc));
        // TODO: CUDNN_CONVOLUTION is a tunable argument
        checkCudnnError(cudnnSetConvolutionNdDescriptor(
            convDesc, numDim - 2, pad, stride, dilation, MODES[record->mode],
            cudnnDataType));
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
        int outDim[numDim];
        checkCudnnError(cudnnGetConvolutionNdForwardOutputDim(
            convDesc, inDesc, knDesc, numDim, outDim));
        cudnnTensorDescriptor_t outDesc;
        checkCudnnError(cudnnCreateTensorDescriptor(&outDesc));
        checkCudnnError(cudnnSetTensorNdDescriptorEx(
            outDesc, CUDNN_TENSOR_NCHW, cudnnDataType, numDim, outDim));
        IT_ASSERT((vector(outDim, outDim + numDim)) ==
                      op->getOutput()->getDims(),
                  "cuDNN output shape mismatches with OP output shape");

        return tuple(inData, knData, outData, inDesc, knDesc, biasDesc,
                     convDesc, actDesc, outDesc);
    }

    bool cuDNNUnfused(const Ref<Conv3dObj> &op,
                      const Conv3dCuDnnPerfRecord &record,
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
        auto record = make_ref<Conv3dCuDnnPerfRecordObj>(); // with paramters in
                                                            // default ctor
        compute(op, record, context);
    }

    PerfRecord tune(const Operator &_op,
                    const RuntimeObj *_context) const override {
        Conv3dCuDnnPerfRecordObj ret;
        ret.time = std::numeric_limits<double>::max();
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        auto op = as<Conv3dObj>(_op);
        // Both modes have the same performance. Only run cross-correlation.
        for (int mode = 1; mode < 2; mode++) {
            // Try every possible algorithm of convolution
            for (int algo = 0; algo < N_ALGO; algo++) {
                auto recordRef = make_ref<Conv3dCuDnnPerfRecordObj>();
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
        return make_ref<Conv3dCuDnnPerfRecordObj>(ret);
    }

    void compute(const Operator &_op, const PerfRecord &_record,
                 const RuntimeObj *_context) const override {
        auto op = as<Conv3dObj>(_op);
        auto record = as<Conv3dCuDnnPerfRecordObj>(_record);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        bool success = cuDNNUnfused(op, record, context);
        IT_ASSERT(success);
    }

    void computeFuncAdd(const Key perfKey, const Operator &op,
                        const PerfRecord &record,
                        const RuntimeObj *context) override {}

    ComputeFuncPtr getComputeFunc(const Key &key) const override {
        return nullptr;
    }

    void setComputeFunc(const Key &key, ComputeFuncPtr ptr) override {}
};

REGISTER_KERNEL(Device::CUDA, OpType::Conv3d, Conv3dCudnn, "Conv3d_cuDNN_CUDA");
} // namespace infini
