#include "operators/conv.h"
#include "core/kernel.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
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

class convCudnn : public Kernel {

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

    std::tuple<void *, void *, void *, void *, cudnnTensorDescriptor_t,
               cudnnFilterDescriptor_t, cudnnTensorDescriptor_t,
               cudnnConvolutionDescriptor_t, cudnnActivationDescriptor_t,
               cudnnTensorDescriptor_t>
    createCuDNNDescriptor(const Ref<ConvObj> &op,
                          const ConvCuDnnPerfRecord &record) const {
        void *const inData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const knData = (op->getInputs(1)->getRawDataPtr<void *>());
        auto cudnnDataType = cudnnDataTypeConvert(op->getDType());
        IT_ASSERT(cudnnDataType == CUDNN_DATA_HALF || cudnnDataType == CUDNN_DATA_FLOAT);

        void *biasData = nullptr;
        if (op->getInputs().size() > 2) {
            IT_ASSERT(op->getInputs()[2]->getDType() == op->getDType(),
                      "Convolution bias must match the output dtype");
            biasData = op->getInputs()[2]->getRawDataPtr<void *>();
        }
        void *const outData = (op->getOutput()->getRawDataPtr<void *>());


        const auto [n, c, h, w, f, r, s] = op->getNCHWFRS();
        IT_ASSERT(op->getInputs(1)->getDType() == op->getDType(),
                  "Convolution weight must match the input dtype");
        if (biasData != nullptr)
            IT_ASSERT(op->getInputs(2)->size() == static_cast<size_t>(f),
                      "Convolution bias must have one value per output channel");
        const int cpg = op->getChannelPerGroup();
        const int g = c / cpg;
        const auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();

        int channelsPerGrp = cpg, channels = c;

        // get inputs
        //  输入张量描述符
        cudnnTensorDescriptor_t inDesc;
        checkCudnnError(cudnnCreateTensorDescriptor(&inDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(
            inDesc, CUDNN_TENSOR_NCHW, cudnnDataType, n, channels, h, w));

        // get kernels
        // 卷积核权重描述符
        cudnnFilterDescriptor_t knDesc;
        checkCudnnError(cudnnCreateFilterDescriptor(&knDesc));
        checkCudnnError(cudnnSetFilter4dDescriptor(
            knDesc, cudnnDataType, CUDNN_TENSOR_NCHW, f, channelsPerGrp, r, s));

        // get bias
        // 偏置描述符
        cudnnTensorDescriptor_t biasDesc;
        checkCudnnError(cudnnCreateTensorDescriptor(&biasDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW,
                                                   cudnnDataType, 1, f, 1, 1));

        // get convolution descriptor
        // 卷积计算描述符
        cudnnConvolutionDescriptor_t convDesc;
        checkCudnnError(cudnnCreateConvolutionDescriptor(&convDesc));
        // TODO: CUDNN_CONVOLUTION is a tunable argument
        checkCudnnError(cudnnSetConvolution2dDescriptor(
            convDesc, ph, pw, sh, sw, dh, dw, MODES[record->mode],
            cudnnDataType)); // Preserve the original fallback compute type.
        if (g > 1) {
            checkCudnnError(cudnnSetConvolutionGroupCount(convDesc, g));
        }
        // tensor core
        if (cudnnDataType == CUDNN_DATA_HALF) {
            checkCudnnError(cudnnSetConvolutionMathType(
                convDesc, CUDNN_TENSOR_OP_MATH));
        }

        // get activation descriptor
        // 激活函数描述符
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
        // 输出描述符
        int outn, outc, outh, outw;
        checkCudnnError(cudnnGetConvolution2dForwardOutputDim(
            convDesc, inDesc, knDesc, &outn, &outc, &outh, &outw));
        cudnnTensorDescriptor_t outDesc;
        checkCudnnError(cudnnCreateTensorDescriptor(&outDesc));
        checkCudnnError(cudnnSetTensor4dDescriptor(
            outDesc, CUDNN_TENSOR_NCHW, cudnnDataType, outn, outc, outh, outw));
        IT_ASSERT((vector{outn, outc, outh, outw}) ==
                      op->getOutput()->getDims(),
                  "cuDNN output shape mismatches with OP output shape");

        //return tuple(inData, knData, outData, inDesc, knDesc, biasDesc,
        //             convDesc, actDesc, outDesc);
        return tuple(inData, knData, outData, biasData,
             inDesc, knDesc, biasDesc, convDesc, actDesc, outDesc);
    }

    bool cuDNNForward(const Ref<ConvObj> &op, const ConvCuDnnPerfRecord &record,
                      const CudaRuntimeObj *context) const {
#ifdef USE_METAX
        const auto &[inData, knData, outData, biasData, inDesc, knDesc, biasDesc,
                     convDesc, actDesc, outDesc] =
            createCuDNNDescriptor(op, record);

        auto destroyDescriptors = [&]() {
            checkCudnnError(cudnnDestroyTensorDescriptor(outDesc));
            checkCudnnError(cudnnDestroyActivationDescriptor(actDesc));
            checkCudnnError(cudnnDestroyConvolutionDescriptor(convDesc));
            checkCudnnError(cudnnDestroyTensorDescriptor(biasDesc));
            checkCudnnError(cudnnDestroyFilterDescriptor(knDesc));
            checkCudnnError(cudnnDestroyTensorDescriptor(inDesc));
        };

        // MACA cu-bridge: query workspace and fall back across algos when the
        // default perf record does not match an executable forward config.
        int tryAlgo[N_ALGO];
        tryAlgo[0] = record->algo;
        int k = 1;
        for (int i = 0; i < N_ALGO; ++i) {
            if (i != record->algo) {
                tryAlgo[k++] = i;
            }
        }

        float alpha1 = 1.f, beta = 0.f, alpha2 = 0.f;

        for (int j = 0; j < N_ALGO; ++j) {
            const int ai = tryAlgo[j];
            size_t wsSize = 0;
            cudnnStatus_t stat = cudnnGetConvolutionForwardWorkspaceSize(
                context->cudnnHandle(), inDesc, knDesc, convDesc, outDesc,
                ALGOS[ai], &wsSize);
            if (stat != CUDNN_STATUS_SUCCESS) {
                continue;
            }
            if (wsSize > context->getWorkspaceSize()) {
                continue;
            }
            void *wsData = context->getWorkspace(wsSize);
            if (biasData != nullptr) {
                stat = cudnnConvolutionBiasActivationForward(
                    context->cudnnHandle(), &alpha1,
                    inDesc, inData,
                    knDesc, knData,
                    convDesc, ALGOS[ai],
                    wsData, wsSize,
                    &alpha2,
                    outDesc, outData,       // 输出
                    biasDesc, biasData,     // bias
                    actDesc, outDesc, outData);  // activation (zDesc === outDesc, zData === outData)
            } else {
                stat = cudnnConvolutionForward(context->cudnnHandle(), &alpha1, inDesc,
                                   inData, knDesc, knData, convDesc,
                                   ALGOS[ai], wsData, wsSize,
                                   &beta, outDesc, outData);
            }
            if (stat == CUDNN_STATUS_SUCCESS) {
                destroyDescriptors();
                return true;
            }
        }

        destroyDescriptors();
        return false;
#else
        const auto &[inData, knData, outData, biasData, inDesc, knDesc, biasDesc,
                     convDesc, actDesc, outDesc] =
            createCuDNNDescriptor(op, record);
        auto destroyDescriptors = [&]() {
            checkCudnnError(cudnnDestroyTensorDescriptor(outDesc));
            checkCudnnError(cudnnDestroyActivationDescriptor(actDesc));
            checkCudnnError(cudnnDestroyConvolutionDescriptor(convDesc));
            checkCudnnError(cudnnDestroyTensorDescriptor(biasDesc));
            checkCudnnError(cudnnDestroyFilterDescriptor(knDesc));
            checkCudnnError(cudnnDestroyTensorDescriptor(inDesc));
        };

        // Scaling parameters are host floats for HALF and FLOAT tensors.
        float alpha1 = 1.f, alpha2 = 0.f, beta = 0.f, betaAdd = 1.f;

        const bool hasBias = biasData != nullptr;
        const bool activationCanFuse = op->getAct() == ActType::None ||
                                       op->getAct() == ActType::Relu;
        const bool canFuse = hasBias && activationCanFuse;

        int tryAlgo[N_ALGO];
        tryAlgo[0] = record->algo;
        int k = 1;
        for (int i = 0; i < N_ALGO; ++i) {
            if (i != record->algo) {
                tryAlgo[k++] = i;
            }
        }

        // Prefer the single-call Conv + Bias + Activation path. Identity is
        // supported only with IMPLICIT_PRECOMP_GEMM by the legacy cuDNN API.
        if (canFuse) {
            for (int j = 0; j < N_ALGO; ++j) {
                const int ai = tryAlgo[j];
                if (op->getAct() == ActType::None &&
                     ALGOS[ai] != CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM) {
                    continue;
                }

                size_t wsSize = 0;
                cudnnStatus_t stat = cudnnGetConvolutionForwardWorkspaceSize(
                    context->cudnnHandle(), inDesc, knDesc, convDesc, outDesc,
                    ALGOS[ai], &wsSize);
                if (stat != CUDNN_STATUS_SUCCESS ||
                    wsSize > context->getWorkspaceSize())
                    continue;

                CudaPtr wsData = context->getWorkspace(wsSize);
                stat = cudnnConvolutionBiasActivationForward(
                    context->cudnnHandle(), &alpha1,
                    inDesc, inData,
                    knDesc, knData,
                    convDesc, ALGOS[ai],
                    wsData, wsSize,
                    &alpha2,
                    outDesc, outData,
                    biasDesc, biasData,     // bias
                    actDesc, outDesc, outData);
                if (stat == CUDNN_STATUS_SUCCESS) {
                    destroyDescriptors();
                    return true;
                }
            }
        }

        for (int j = 0; j < N_ALGO; ++j) {
            const int ai = tryAlgo[j];
            size_t wsSize = 0;

            cudnnStatus_t stat = cudnnGetConvolutionForwardWorkspaceSize(
                context->cudnnHandle(), inDesc, knDesc, convDesc, outDesc,
                ALGOS[ai], &wsSize);
            if (stat != CUDNN_STATUS_SUCCESS)
                continue;
            if (wsSize > context->getWorkspaceSize()) {
                continue;
            }

            CudaPtr wsData = context->getWorkspace(wsSize);

            stat = cudnnConvolutionForward(
                context->cudnnHandle(), &alpha1, inDesc, inData,
                knDesc, knData, convDesc, ALGOS[ai],
                wsData, wsSize, &beta, outDesc, outData);
            if (stat != CUDNN_STATUS_SUCCESS)
                continue;
            if (biasData != nullptr) {
                stat = cudnnAddTensor(context->cudnnHandle(), &alpha1, biasDesc,
                                      biasData, &betaAdd, outDesc, outData);
                if (stat != CUDNN_STATUS_SUCCESS)
                    break;
            }
            if (op->getAct() != ActType::None) {
                stat = cudnnActivationForward(context->cudnnHandle(), actDesc,
                                              &alpha1, outDesc, outData, &beta,
                                              outDesc, outData);
                if (stat != CUDNN_STATUS_SUCCESS)
                    break;
            }
            destroyDescriptors();
            return true;
        }

        destroyDescriptors();
        return false;
#endif
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
        IT_ASSERT(context != nullptr);
        if (convNhwcEnabled() && convNhwcEligible(op)) {
            auto &cache = context->getConvNhwcCache();
            size_t workspaceBytes = 0;
            // First call builds the plan and pre-packs weights, outside timing.
            if (cache.run(op, 1, context, &workspaceBytes)) {
                context->sync();
                ConvCuDnnPerfRecordObj nhwcRecord;
                nhwcRecord.mode = 1;
                nhwcRecord.algo = 1;
                // Record the preferred fusion; the cache may use ordinary
                // NHWC forward if the fused call is unsupported.
                nhwcRecord.fuseAct = op->getInputs().size() == 3 &&
                                     op->getAct() == ActType::None;
                nhwcRecord.workspaceSize = workspaceBytes;
                nhwcRecord.time = timeit(
                    [&]() {
                        IT_ASSERT(cache.run(op, 1, context),
                                  "NHWC path failed during tuning");
                    },
                    [&]() { context->sync(); });
                return make_ref<ConvCuDnnPerfRecordObj>(nhwcRecord);
            }
        }
        // Both modes have the same performance. Only run cross-correlation.
        for (int mode = 1; mode < 2; mode++) {
            // Try every possible algorithm of convolution
            for (int algo = 0; algo < N_ALGO; algo++) {
                auto recordRef = make_ref<ConvCuDnnPerfRecordObj>();
                auto &record = *recordRef;
                record.mode = mode;
                record.algo = algo;
                cudnnStatus_t stat;
                const auto &[inData, knData, outData, biasData, inDesc, knDesc, biasDesc,
                             convDesc, actDesc, outDesc] =
                    createCuDNNDescriptor(op, recordRef);
                auto destroyDescriptors = [&]() {
                    checkCudnnError(cudnnDestroyTensorDescriptor(outDesc));
                    checkCudnnError(cudnnDestroyActivationDescriptor(actDesc));
                    checkCudnnError(cudnnDestroyConvolutionDescriptor(convDesc));
                    checkCudnnError(cudnnDestroyTensorDescriptor(biasDesc));
                    checkCudnnError(cudnnDestroyFilterDescriptor(knDesc));
                    checkCudnnError(cudnnDestroyTensorDescriptor(inDesc));
                };

                // get workspace
                stat = cudnnGetConvolutionForwardWorkspaceSize(
                    context->cudnnHandle(), inDesc, knDesc, convDesc, outDesc,
                    ALGOS[record.algo], &record.workspaceSize);
                if (stat != CUDNN_STATUS_SUCCESS) {
                    destroyDescriptors();
                    continue;
                }
                if (record.workspaceSize > context->getWorkspaceSize()) {
                    destroyDescriptors();
                    continue;
                }
                CudaPtr wsData = context->getWorkspace(record.workspaceSize);

                float alpha = 1.f, beta = 0.f;
                stat = cudnnConvolutionForward(context->cudnnHandle(), &alpha,
                                               inDesc, inData, knDesc, knData,
                                               convDesc, ALGOS[record.algo],
                                               wsData, record.workspaceSize,
                                               &beta, outDesc, outData);

                if (stat != CUDNN_STATUS_SUCCESS) {
                    destroyDescriptors();
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

                // Update the tune result
                if (ret.time > record.time) {
                    ret = record;
                }
                destroyDescriptors();
            }
        }
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
        IT_ASSERT(context != nullptr);
        if (convNhwcEnabled() && convNhwcEligible(op)) {
            auto &cache = context->getConvNhwcCache();
            if (cache.run(op, record->mode, context)) {
                return;
            }
        }
        bool success = cuDNNForward(op, record, context);
        IT_ASSERT(success);
    }

  public:
    convCudnn() {
        auto computePtr = [this](const Operator &op, const PerfRecord &record,
                                 const RuntimeObj *context) {
            this->compute(op, record, context);
        };
        funcVec.emplace_back(computePtr);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Conv, convCudnn, "Conv_cuDNN_CUDA");

REGISTER_CONSTRUCTOR(1, ConvCuDnnPerfRecordObj::from_json);
} // namespace infini
