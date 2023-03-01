#include "core/kernel.h"
#include "cuda/cuda_runtime.h"
#include "operators/conv.h"
#include <chrono>
#include <functional>
#include <limits>
#include <tuple>
namespace infini {

struct ConvTransposedCuDnnPerfRecordObj : public PerfRecordObj {
    int algo = 0; // cudnnConvolutionBwdDataAlgo_t
    int mode = 1;
    size_t workspaceSize = 100000;
    bool fuseAct = false;
};
using ConvTransposedCuDnnPerfRecord = Ref<ConvTransposedCuDnnPerfRecordObj>;

static constexpr int N_ALGO = 6;
static_assert(N_ALGO == int(CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT),
              "Unsupported cuDNN version");
static const cudnnConvolutionBwdDataAlgo_t ALGOS[N_ALGO] = {
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, /* non-deterministic */
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED};
static const char algo_name[N_ALGO][50] = {
    // only first two can be used for NHWC format
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_0", /* non-deterministic */
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_1",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED"};
static const char math_types[3][50] = {"CUDNN_DEFAULT_MATH",
                                       "CUDNN_TENSOR_OP_MATH",
                                       "CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION"};
static constexpr int N_MODE = 2;
static constexpr cudnnConvolutionMode_t MODES[N_MODE] = {
    CUDNN_CONVOLUTION, CUDNN_CROSS_CORRELATION};

class convBackwardDataCudnn : public Kernel {

    std::tuple<void *, void *, void *, cudnnTensorDescriptor_t,
               cudnnFilterDescriptor_t, cudnnTensorDescriptor_t,
               cudnnConvolutionDescriptor_t, cudnnActivationDescriptor_t,
               cudnnTensorDescriptor_t>
    createCuDNNDescriptor(
        const Ref<ConvBaseObj> &op,
        const ConvTransposedCuDnnPerfRecordObj &record) const {
        void *const inData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const knData = (op->getInputs(1)->getRawDataPtr<void *>());
        if (op->getInputs().size() > 2) // Bias is not supported yet
            IT_TODO_HALT();
        // void *const biasData = (op->getInputs(2)->getRawDataPtr<void
        // *>());
        void *const outData = (op->getOutput()->getRawDataPtr<void *>());

        const auto [n, c, h, w, f, r, s] = op->getNCHWFRS();
        const int channelsPerGrp = op->getChannelPerGroup();
        const int g = op->getNumGroups();
        const auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
        // IT_ASSERT(g == 1, "Group convolution is not supported yet");

        // set input format
        cudnnTensorFormat_t tensorFormat =
            (op->getOpType() == OpType::ConvTransNHWC) ? CUDNN_TENSOR_NHWC
                                                       : CUDNN_TENSOR_NCHW;

        // get inputs
        cudnnTensorDescriptor_t inDesc;
        checkCudnnError(cudnnCreateTensorDescriptor(&inDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(
            inDesc, tensorFormat, CUDNN_DATA_FLOAT, n, f, h, w));

        // get kernels
        cudnnFilterDescriptor_t knDesc;
        checkCudnnError(cudnnCreateFilterDescriptor(&knDesc));
        checkCudnnError(cudnnSetFilter4dDescriptor(
            knDesc, CUDNN_DATA_FLOAT, tensorFormat, f, channelsPerGrp, r, s));
        // get bias
        cudnnTensorDescriptor_t biasDesc;
        checkCudnnError(cudnnCreateTensorDescriptor(&biasDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(
            biasDesc, tensorFormat, CUDNN_DATA_FLOAT, 1, f, 1, 1));

        // get convlution descriptor
        cudnnConvolutionDescriptor_t convDesc;
        checkCudnnError(cudnnCreateConvolutionDescriptor(&convDesc));
        // TODO: CUDNN_CONVOLUTION is a tunable argument
        checkCudnnError(cudnnSetConvolution2dDescriptor(
            convDesc, ph, pw, sh, sw, dh, dw, MODES[record.mode],
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

        const auto &outputShape = op->getOutput()->getDims();
        int on, oh, ow, oc;
        if (op->getOpType() == OpType::ConvTransNHWC) {
            on = outputShape[0];
            oh = outputShape[1];
            ow = outputShape[2];
            oc = outputShape[3];
        } else {
            on = outputShape[0];
            oh = outputShape[2];
            ow = outputShape[3];
            oc = outputShape[1];
        }
        cudnnTensorDescriptor_t outDesc;
        checkCudnnError(cudnnCreateTensorDescriptor(&outDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(
            outDesc, tensorFormat, CUDNN_DATA_FLOAT, on, oc, oh, ow));
        return tuple(inData, knData, outData, inDesc, knDesc, biasDesc,
                     convDesc, actDesc, outDesc);
    }

    bool cuDNNUnfused(const Ref<ConvBaseObj> &op,
                      const ConvTransposedCuDnnPerfRecordObj &record,
                      const CudaRuntimeObj *context) const {
        cudnnStatus_t stat;

        const auto &[inData, knData, outData, inDesc, knDesc, biasDesc,
                     convDesc, actDesc, outDesc] =
            createCuDNNDescriptor(op, record);
        size_t wsSize = record.workspaceSize;
        CudaPtr wsData = context->getWorkspace(wsSize);
        float alpha = 1.f, beta = 0.f;

        stat = cudnnConvolutionBackwardData(
            context->cudnnHandle(), &alpha, knDesc, knData, inDesc, inData,
            convDesc, ALGOS[record.algo], wsData, wsSize, &beta, outDesc,
            outData);
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
        //         cudnnHandle(), &alpha, inDesc, inData, knDesc,
        //         knData, convDesc, ALGOS[i], wsData, wsSize, &beta,
        //         outDesc, outData, biasDesc, biasData, actDesc,
        //         outDesc, outData);
        //     if (stat != CUDNN_STATUS_SUCCESS) {
        //         // checkCudnnError(stat);
        //         // Do not checkCudnnError since not all algorithms
        //         are
        //         // supported
        //         durtime_fuse = INFINITY;
        //         break;
        //     }
        // }

        // Destories in CUDA does not require sync. But cuDNN does not
        // state whether sync is required before destories.
        checkCudnnError(cudnnDestroyTensorDescriptor(outDesc));
        checkCudnnError(cudnnDestroyActivationDescriptor(actDesc));
        checkCudnnError(cudnnDestroyConvolutionDescriptor(convDesc));
        checkCudnnError(cudnnDestroyTensorDescriptor(biasDesc));
        checkCudnnError(cudnnDestroyFilterDescriptor(knDesc));
        checkCudnnError(cudnnDestroyTensorDescriptor(inDesc));
        return true;
    }

    void compute(const Operator &op, const RuntimeObj *context) const override {
        // with paramters in default ctor
        auto record = make_ref<ConvTransposedCuDnnPerfRecordObj>();
        compute(op, record, context);
    }

    PerfRecord tune(const Operator &_op,
                    const RuntimeObj *_context) const override {
        ConvTransposedCuDnnPerfRecordObj ret;
        ret.time = std::numeric_limits<double>::max();
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        auto op = as<ConvBaseObj>(_op);
        // Both modes have the same performance. Only run
        // cross-correlation.
        int algo_to_run =
            (op->getOpType() == OpType::ConvTransNHWC) ? 2 : N_ALGO;
        for (int mode = 1; mode < 2; mode++) {
            // Try every possible algorithm of convolution
            for (int algo = 0; algo < algo_to_run; algo++) {
                ConvTransposedCuDnnPerfRecordObj record;
                record.mode = mode;
                record.algo = algo;
                cudnnStatus_t stat;
                const auto &[inData, knData, outData, inDesc, knDesc, biasDesc,
                             convDesc, actDesc, outDesc] =
                    createCuDNNDescriptor(op, record);

                // get workspace
                stat = cudnnGetConvolutionBackwardDataWorkspaceSize(
                    context->cudnnHandle(), knDesc, inDesc, convDesc, outDesc,
                    ALGOS[record.algo], &record.workspaceSize);
                if (stat != CUDNN_STATUS_SUCCESS)
                    continue;

                CudaPtr wsData = context->getWorkspace(record.workspaceSize);
                float alpha = 1.f, beta = 0.f;

                stat = cudnnConvolutionBackwardData(
                    context->cudnnHandle(), &alpha, knDesc, knData, inDesc,
                    inData, convDesc, ALGOS[record.algo], wsData,
                    record.workspaceSize, &beta, outDesc, outData);
                if (stat != CUDNN_STATUS_SUCCESS)
                    continue;
                record.time = timeit(
                    [&]() {
                        cudnnConvolutionBackwardData(
                            context->cudnnHandle(), &alpha, knDesc, knData,
                            inDesc, inData, convDesc, ALGOS[record.algo],
                            wsData, record.workspaceSize, &beta, outDesc,
                            outData);
                    },
                    [&]() { context->sync(); });
                // printf("mode:%d algo:%d :%.8lf\n", mode, algo, record.time);

                // Update the tune result
                if (ret.time > record.time)
                    ret = record;
                checkCudnnError(cudnnDestroyTensorDescriptor(outDesc));
                checkCudnnError(cudnnDestroyActivationDescriptor(actDesc));
                checkCudnnError(cudnnDestroyConvolutionDescriptor(convDesc));
                checkCudnnError(cudnnDestroyTensorDescriptor(biasDesc));
                checkCudnnError(cudnnDestroyFilterDescriptor(knDesc));
                checkCudnnError(cudnnDestroyTensorDescriptor(inDesc));
            }
        }
        // printf("the best algo is %d, the best conv mode is %d\n",
        // ret.algo,
        //        ret.mode);
        IT_ASSERT(ret.time < std::numeric_limits<double>::max(), "No valid "
                                                                 "algorithm "
                                                                 "found");
        return make_ref<ConvTransposedCuDnnPerfRecordObj>(ret);
    }

    void compute(const Operator &_op, const PerfRecord &_record,
                 const RuntimeObj *_context) const override {
        auto op = as<ConvBaseObj>(_op);
        auto record = as<ConvTransposedCuDnnPerfRecordObj>(_record);
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        bool success = cuDNNUnfused(op, *record, context);
        IT_ASSERT(success);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::ConvTrans, DataType::Float32,
                convBackwardDataCudnn, "ConvTranposed_cuDNN_CUDA_Float32");
REGISTER_KERNEL(Device::CUDA, OpType::ConvTransNHWC, DataType::Float32,
                convBackwardDataCudnn, "ConvTranposedNHWC_cuDNN_CUDA_Float32");
} // namespace infini
