#include "cuda/conv_nhwc_cache.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_unary.h"
#include "operators/conv.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

namespace infini {

bool convNhwcEnabled() {
    const char *value = std::getenv("INFINITENSOR_CONV_NHWC");
    // Shared CUDA/CoreX/MetaX path. Unsupported library plans fall back to NCHW.
    return !value || std::strcmp(value, "0") != 0;
}

bool convNhwcRegionEnabled() {
    const char *value = std::getenv("INFINITENSOR_CONV_NHWC_REGION");
    return convNhwcEnabled() && (!value || std::strcmp(value, "0") != 0);
}

bool convNhwcEligible(const Ref<ConvObj> &op) {
    if (!(op->getDType() == DataType::Float16) ||
        (op->getInputs().size() != 2 && op->getInputs().size() != 3) ||
        op->getInputs(0)->getRank() != 4 || op->getOutput()->getRank() != 4)
        return false;
    if (op->getAct() != ActType::None && op->getAct() != ActType::Relu &&
        op->getAct() != ActType::Sigmoid)
        return false;

    const auto [n, c, h, w, f, r, s] = op->getNCHWFRS();
    const auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
    const auto weight = op->getInputs(1);
    const auto bias = op->getInputs().size() == 3 ? op->getInputs(2) : nullptr;
    // Keep C=3, grouped, dilated and dynamic-weight convolutions on the
    // existing path. NHWC + tensor-op math does not guarantee TCU dispatch.
    return n > 0 && c > 0 && h > 0 && w > 0 && f > 0 && r > 0 && s > 0 &&
           op->getChannelPerGroup() == c && dh == 1 && dw == 1 && c % 8 == 0 &&
           f % 8 == 0 && weight->isWeight() && !weight->getSource() &&
           weight->getDType() == DataType::Float16 &&
           (!bias || (bias->getDType() == DataType::Float16 &&
                      bias->size() == static_cast<size_t>(f)));
}

namespace {
bool regionOp(const Operator &op) {
    if (op->getOpType() == OpType::Conv)
        return convNhwcEligible(as<ConvObj>(op));
    return op->getOpType() == OpType::Silu && op->getInputs().size() == 1 &&
           op->getOutputs().size() == 1 &&
           op->getDType() == DataType::Float16 &&
           op->getOutput()->getDType() == DataType::Float16 &&
           op->getInputs(0)->getRank() == 4 &&
           op->getInputs(0)->getDims() == op->getOutput()->getDims();
}

bool canFollow(const Operator &producer, const Operator &consumer) {
    if (!regionOp(producer) || !regionOp(consumer))
        return false;
    const auto output = producer->getOutput();
    const auto targets = output->getTargets();
    return output->isOthers() && targets.size() == 1 &&
           targets.front() == consumer && consumer->getInputs(0) == output;
}
} // namespace

std::vector<OpVec> findConvNhwcRegions(const OpVec &ops) {
    const std::unordered_set<Operator> members(ops.begin(), ops.end());
    std::unordered_set<Operator> assigned;
    std::vector<OpVec> regions;
    for (const auto &op : ops) {
        if (assigned.count(op) || op->getOpType() != OpType::Conv ||
            !regionOp(op))
            continue;
        OpVec region{op};
        size_t convs = 1;
        while (true) {
            const auto targets = region.back()->getOutput()->getTargets();
            if (targets.size() != 1 || !members.count(targets.front()) ||
                assigned.count(targets.front()) ||
                !canFollow(region.back(), targets.front()))
                break;
            region.push_back(targets.front());
            convs += targets.front()->getOpType() == OpType::Conv;
        }
        // ponytail: linear Conv/SiLU only; branched regions need separate
        // live tensors, rather than two shared ping-pong buffers.
        if (convs >= 2) {
            assigned.insert(region.begin(), region.end());
            regions.push_back(std::move(region));
        }
    }
    return regions;
}

bool scheduleConvNhwcRegions(OpVec &ops) {
    if (!convNhwcRegionEnabled())
        return false;
    const auto regions = findConvNhwcRegions(ops);
    std::unordered_map<Operator, size_t> owner;
    for (size_t i = 0; i < regions.size(); ++i)
        for (const auto &op : regions[i])
            owner.emplace(op, i);
    OpVec scheduled;
    scheduled.reserve(ops.size());
    for (const auto &op : ops) {
        const auto it = owner.find(op);
        if (it == owner.end())
            scheduled.push_back(op);
        else if (op == regions[it->second].back()) {
            // Delay the region to its last node's original position. Internal
            // outputs have no external users; all external inputs (including
            // later Conv biases) are ready here. Reallocate after reordering.
            const auto &region = regions[it->second];
            scheduled.insert(scheduled.end(), region.begin(), region.end());
        }
    }
    if (scheduled == ops)
        return false;
    ops = std::move(scheduled);
    return true;
}

struct ConvNhwcCache::Impl {
    struct Plan {
        WRef<ConvObj> owner;
        WRef<BlobObj> sourceBlob;
        std::vector<int> signature;
        const void *sourceWeight = nullptr;
        bool disabled = false;
        bool fuseBias = false;
        size_t workspaceBytes = 0;
        void *packedWeight = nullptr;

        cudnnTensorDescriptor_t xN = nullptr, xH = nullptr;
        cudnnTensorDescriptor_t yN = nullptr, yH = nullptr;
        cudnnTensorDescriptor_t wN = nullptr, wH = nullptr;
        cudnnTensorDescriptor_t bias = nullptr;
        cudnnFilterDescriptor_t filter = nullptr;
        cudnnConvolutionDescriptor_t conv = nullptr;
        cudnnActivationDescriptor_t act = nullptr;
        std::vector<cudnnTensorDescriptor_t> tensors;

        ~Plan() {
            // Free first: previously queued work may still use the descriptors.
            if (packedWeight)
                cudaFree(packedWeight);
            for (auto desc : tensors)
                cudnnDestroyTensorDescriptor(desc);
            if (filter)
                cudnnDestroyFilterDescriptor(filter);
            if (act)
                cudnnDestroyActivationDescriptor(act);
            if (conv)
                cudnnDestroyConvolutionDescriptor(conv);
        }
    };

    int device;
    std::map<UidBaseType, std::unique_ptr<Plan>> plans;
    void *xScratch = nullptr, *yScratch = nullptr;
    size_t xCapacity = 0, yCapacity = 0;
    size_t weightPrepacks = 0, submissions = 0;
    size_t regions = 0, regionConvSubmissions = 0, regionSiluSubmissions = 0;
    size_t inputTransforms = 0, outputTransforms = 0;

    explicit Impl(int device) : device(device) {}

    ~Impl() {
        int previousDevice = device;
        const bool restore = cudaGetDevice(&previousDevice) == cudaSuccess;
        cudaSetDevice(device);
        cudaDeviceSynchronize();
        plans.clear();
        if (xScratch)
            cudaFree(xScratch);
        if (yScratch)
            cudaFree(yScratch);
        if (restore)
            cudaSetDevice(previousDevice);
    }

    static bool usable(cudnnStatus_t status) {
        if (status == CUDNN_STATUS_SUCCESS)
            return true;
        if (status == CUDNN_STATUS_NOT_SUPPORTED ||
            status == CUDNN_STATUS_BAD_PARAM)
            return false;
        // Execution/asynchronous GPU failures are not recoverable fallbacks.
        checkCudnnError(status);
        return false;
    }

    static cudnnTensorDescriptor_t tensor(Plan &p, cudnnTensorFormat_t format,
                                          int n, int c, int h, int w) {
        cudnnTensorDescriptor_t desc;
        checkCudnnError(cudnnCreateTensorDescriptor(&desc));
        p.tensors.push_back(desc);
        if (!usable(cudnnSetTensor4dDescriptor(
                desc, format, CUDNN_DATA_HALF, n, c, h, w)))
            return nullptr;
        return desc;
    }

    static void reserve(void *&ptr, size_t &capacity, size_t bytes) {
        if (bytes <= capacity)
            return;
        void *newPtr = nullptr;
        checkCudaError(cudaMalloc(&newPtr, bytes));
        if (ptr) {
            // cudaFree can synchronize; growth is restricted to non-capture
            // setup/warmup, never a steady-state inference allocation.
            auto status = cudaFree(ptr);
            if (status != cudaSuccess) {
                cudaFree(newPtr);
                checkCudaError(status);
            }
        }
        ptr = newPtr;
        capacity = bytes;
    }

    static std::vector<int> signature(const Ref<ConvObj> &op, int mode) {
        const auto [n, c, h, w, f, r, s] = op->getNCHWFRS();
        const auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
        const auto y = op->getOutput()->getDims();
        return {n, c, h, w, f, r, s, op->getChannelPerGroup(),
                ph, pw, sh, sw, dh, dw, mode, y[0], y[1], y[2], y[3],
                int(op->getInputs().size() == 3), int(op->getAct())};
    }

    bool prepare(Plan &p, const Ref<ConvObj> &op, int mode,
                 const CudaRuntimeObj *context, cudaStream_t stream) {
        const auto [n, c, h, w, f, r, s] = op->getNCHWFRS();
        const auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
        const auto y = op->getOutput()->getDims();
        const int cpg = op->getChannelPerGroup();

        p.xN = tensor(p, CUDNN_TENSOR_NCHW, n, c, h, w);
        p.xH = tensor(p, CUDNN_TENSOR_NHWC, n, c, h, w);
        p.yN = tensor(p, CUDNN_TENSOR_NCHW, y[0], y[1], y[2], y[3]);
        p.yH = tensor(p, CUDNN_TENSOR_NHWC, y[0], y[1], y[2], y[3]);
        // Logical dimensions remain F,C/group,R,S; physical storage is FRSC.
        p.wN = tensor(p, CUDNN_TENSOR_NCHW, f, cpg, r, s);
        p.wH = tensor(p, CUDNN_TENSOR_NHWC, f, cpg, r, s);
        if (!p.xN || !p.xH || !p.yN || !p.yH || !p.wN || !p.wH)
            return false;
        const bool hasBias = op->getInputs().size() == 3;
        if (hasBias) {
            p.bias = tensor(p, CUDNN_TENSOR_NHWC, 1, f, 1, 1);
            if (!p.bias)
                return false;
        }
        p.fuseBias = hasBias && op->getAct() == ActType::None;
        checkCudnnError(cudnnCreateFilterDescriptor(&p.filter));
        if (!usable(cudnnSetFilter4dDescriptor(
                p.filter, CUDNN_DATA_HALF, CUDNN_TENSOR_NHWC, f, cpg, r, s)))
            return false;
        checkCudnnError(cudnnCreateConvolutionDescriptor(&p.conv));
        if (!usable(cudnnSetConvolution2dDescriptor(
                p.conv, ph, pw, sh, sw, dh, dw,
                mode == 0 ? CUDNN_CONVOLUTION : CUDNN_CROSS_CORRELATION,
                CUDNN_DATA_FLOAT)) ||
            !usable(cudnnSetConvolutionGroupCount(p.conv, c / cpg)) ||
            !usable(cudnnSetConvolutionMathType(p.conv, CUDNN_TENSOR_OP_MATH)))
            return false;
        if (p.fuseBias || op->getAct() != ActType::None) {
            checkCudnnError(cudnnCreateActivationDescriptor(&p.act));
            auto activation = CUDNN_ACTIVATION_IDENTITY;
            if (op->getAct() == ActType::Relu)
                activation = CUDNN_ACTIVATION_RELU;
            else if (op->getAct() == ActType::Sigmoid)
                activation = CUDNN_ACTIVATION_SIGMOID;
            if (!usable(cudnnSetActivationDescriptor(
                    p.act, activation, CUDNN_NOT_PROPAGATE_NAN, 0)))
                return false;
        }

        auto status = cudnnGetConvolutionForwardWorkspaceSize(
            context->cudnnHandle(), p.xH, p.filter, p.conv, p.yH,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
            &p.workspaceBytes);
        if (!usable(status) || p.workspaceBytes > context->getWorkspaceSize())
            return false;

        checkCudaError(cudaMalloc(&p.packedWeight, op->getInputs(1)->getBytes()));
        const float one = 1.f, zero = 0.f;
        status = cudnnTransformTensor(context->cudnnHandle(), &one,
                                      p.wN, p.sourceWeight, &zero,
                                      p.wH, p.packedWeight);
        if (!usable(status))
            return false;
        // Once per plan, not per inference. Also makes the immutable packed
        // weight ready if a later CUDA Graph uses a different stream.
        checkCudaError(cudaStreamSynchronize(stream));
        ++weightPrepacks;
        return true;
    }

    Plan &getPlan(const Ref<ConvObj> &op, int mode,
                  const CudaRuntimeObj *context, cudaStream_t stream,
                  cudaStreamCaptureStatus capture) {
        const auto weight = op->getInputs(1);
        const auto sig = signature(op, mode);
        const auto blob = weight->getDataBlob();
        const void *source = weight->getRawDataPtr<void *>();
        const auto key = op->getGuid();
        auto it = plans.find(key);
        const bool rebuild = it == plans.end() ||
                             it->second->owner.lock().get() != op.get() ||
                             it->second->signature != sig ||
                             it->second->sourceWeight != source ||
                             it->second->sourceBlob.lock() != blob;
        if (rebuild) {
            IT_ASSERT(capture == cudaStreamCaptureStatusNone,
                      "Warm up ordinary run before NHWC CUDA Graph capture");
            IT_ASSERT(!context->hasCapturedCudaGraph(),
                      "Recreate runtime before replacing captured NHWC plans");
            for (auto old = plans.begin(); old != plans.end();) {
                if (old->second->owner.expired())
                    old = plans.erase(old);
                else
                    ++old;
            }
            auto p = std::make_unique<Plan>();
            p->owner = op;
            p->sourceBlob = blob;
            p->sourceWeight = source;
            p->signature = sig;
            p->disabled = !prepare(*p, op, mode, context, stream);
            plans[key] = std::move(p);
        }
        return *plans.at(key);
    }

    void reserveScratch(size_t xBytes, size_t yBytes,
                        const CudaRuntimeObj *context,
                        cudaStreamCaptureStatus capture) {
        const bool fits = xBytes <= xCapacity && yBytes <= yCapacity;
        IT_ASSERT(capture == cudaStreamCaptureStatusNone || fits,
                  "Warm up all shapes before NHWC CUDA Graph capture");
        IT_ASSERT(
            !context->hasCapturedCudaGraph() || fits,
            "Recreate runtime before growing captured NHWC scratch buffers");
        reserve(xScratch, xCapacity, xBytes);
        reserve(yScratch, yCapacity, yBytes);
    }

    bool compute(Plan &p, const Ref<ConvObj> &op, const CudaRuntimeObj *context,
                 const void *input, void *output) {
        const float one = 1.f, zero = 0.f;
        const auto handle = context->cudnnHandle();
        const void *bias = op->getInputs().size() == 3
                               ? op->getInputs(2)->getRawDataPtr<void *>()
                               : nullptr;
        void *workspace = p.workspaceBytes
                              ? context->getWorkspace(p.workspaceBytes)
                              : nullptr;
        cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
        if (p.fuseBias) {
            status = cudnnConvolutionBiasActivationForward(
                handle, &one, p.xH, input, p.filter, p.packedWeight, p.conv,
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, workspace,
                p.workspaceBytes, &zero, p.yH, output, p.bias, bias, p.act,
                p.yH, output);
            // Remember a rejected fusion, but retain the ordinary NHWC path.
            if (!usable(status))
                p.fuseBias = false;
        }
        if (!p.fuseBias) {
            status = cudnnConvolutionForward(
                handle, &one, p.xH, input, p.filter, p.packedWeight, p.conv,
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, workspace,
                p.workspaceBytes, &zero, p.yH, output);
            if (usable(status) && bias)
                status = cudnnAddTensor(handle, &one, p.bias, bias, &one, p.yH,
                                        output);
            if (usable(status) && op->getAct() != ActType::None)
                status = cudnnActivationForward(handle, p.act, &one, p.yH,
                                                output, &zero, p.yH, output);
        }
        if (!usable(status)) {
            p.disabled = true;
            return false;
        }
        ++submissions;
        return true;
    }
};

ConvNhwcCache::ConvNhwcCache(int device)
    : impl(std::make_unique<Impl>(device)) {}
ConvNhwcCache::~ConvNhwcCache() = default;

ConvNhwcCacheStats ConvNhwcCache::stats() const {
    return {impl->plans.size(),          impl->weightPrepacks,
            impl->submissions,           impl->regions,
            impl->regionConvSubmissions, impl->regionSiluSubmissions,
            impl->inputTransforms,       impl->outputTransforms};
}

bool ConvNhwcCache::run(const Ref<ConvObj> &op, int mode,
                        const CudaRuntimeObj *context, size_t *workspaceBytes) {
    if (!convNhwcEligible(op))
        return false;
    IT_ASSERT(mode == 0 || mode == 1);
    int currentDevice;
    checkCudaError(cudaGetDevice(&currentDevice));
    IT_ASSERT(currentDevice == impl->device,
              "Active CUDA device must match runtime");
    cudaStream_t stream;
    checkCudnnError(cudnnGetStream(context->cudnnHandle(), &stream));
    cudaStreamCaptureStatus capture;
    checkCudaError(cudaStreamIsCapturing(stream, &capture));

    auto &p = impl->getPlan(op, mode, context, stream, capture);
    if (p.disabled)
        return false;
    impl->reserveScratch(op->getInputs(0)->getBytes(),
                         op->getOutput()->getBytes(), context, capture);

    const float one = 1.f, zero = 0.f;
    const auto handle = context->cudnnHandle();
    const void *input = op->getInputs(0)->getRawDataPtr<void *>();
    void *output = op->getOutput()->getRawDataPtr<void *>();
    auto status = cudnnTransformTensor(handle, &one, p.xN, input, &zero, p.xH,
                                       impl->xScratch);
    if (!Impl::usable(status)) {
        p.disabled = true;
        return false;
    }
    ++impl->inputTransforms;
    if (!impl->compute(p, op, context, impl->xScratch, impl->yScratch))
        return false;
    status = cudnnTransformTensor(handle, &one, p.yH, impl->yScratch, &zero,
                                  p.yN, output);
    if (!Impl::usable(status)) {
        p.disabled = true;
        return false;
    }
    if (workspaceBytes)
        *workspaceBytes = p.workspaceBytes;
    ++impl->outputTransforms;
    const char *debug = std::getenv("INFINITENSOR_CONV_DEBUG");
    if (capture == cudaStreamCaptureStatusNone && debug &&
        std::strcmp(debug, "1") == 0) {
        const auto [n, c, h, w, f, r, s] = op->getNCHWFRS();
        std::fprintf(stderr,
                     "[Conv] path=%s dtype=Float16 compute=Float32 "
                     "algo=IMPLICIT_PRECOMP_GEMM bias=%d act=%d "
                     "n=%d c=%d h=%d w=%d f=%d r=%d s=%d\n",
                     p.fuseBias ? "NHWC_FUSED" : "NHWC_FORWARD",
                     op->getInputs().size() == 3, int(op->getAct()), n, c, h, w,
                     f, r, s);
    }
    return true;
}

size_t ConvNhwcCache::runRegion(const OpVec &ops, size_t begin,
                                const CudaRuntimeObj *context) {
    if (!convNhwcRegionEnabled() || begin >= ops.size() ||
        ops[begin]->getOpType() != OpType::Conv || !regionOp(ops[begin]))
        return 0;
    // optimize() has scheduled data-connected regions contiguously and
    // rebuilt their memory lifetimes. Recheck edges for ordinary unoptimized
    // graphs and for callers that changed tensor metadata after optimization.
    size_t end = begin + 1;
    while (end < ops.size() && canFollow(ops[end - 1], ops[end]))
        ++end;
    if (end - begin < 2)
        return 0;

    int currentDevice;
    checkCudaError(cudaGetDevice(&currentDevice));
    IT_ASSERT(currentDevice == impl->device,
              "Active CUDA device must match runtime");
    cudaStream_t stream;
    const auto handle = context->cudnnHandle();
    checkCudnnError(cudnnGetStream(handle, &stream));
    cudaStreamCaptureStatus capture;
    checkCudaError(cudaStreamIsCapturing(stream, &capture));
    std::vector<Impl::Plan *> plans;
    size_t bytes = 0, convs = 0;
    Impl::Plan *lastConv = nullptr;
    // Prepare the entire supported prefix before enqueuing any region work.
    // Weight packing and buffer growth are forbidden during graph capture.
    for (size_t i = begin; i < end; ++i) {
        const auto &op = ops[i];
        Impl::Plan *plan = nullptr;
        if (op->getOpType() == OpType::Conv) {
            plan = &impl->getPlan(as<ConvObj>(op), 1, context, stream, capture);
            if (plan->disabled)
                break;
            lastConv = plan;
            ++convs;
        }
        plans.push_back(plan);
        bytes = std::max(
            {bytes, op->getInputs(0)->getBytes(), op->getOutput()->getBytes()});
    }
    if (convs < 2)
        return 0;
    // Fixed ping-pong addresses, both large enough for every layer. Swapping
    // only local pointers preserves the addresses held by captured graphs.
    impl->reserveScratch(bytes, bytes, context, capture);
    const float one = 1.f, zero = 0.f;
    const auto first = as<ConvObj>(ops[begin]);
    auto status =
        cudnnTransformTensor(handle, &one, plans.front()->xN,
                             first->getInputs(0)->getRawDataPtr<void *>(),
                             &zero, plans.front()->xH, impl->xScratch);
    if (!Impl::usable(status)) {
        plans.front()->disabled = true;
        return 0;
    }
    ++impl->inputTransforms;
    ++impl->regions;
    void *input = impl->xScratch, *output = impl->yScratch;
    for (size_t i = 0; i < plans.size(); ++i) {
        const auto &op = ops[begin + i];
        if (!plans[i]) {
            silu_kernel_fp16(input, output, op->getOutput()->size(), stream);
            checkCudaError(cudaGetLastError());
            ++impl->regionSiluSubmissions;
            std::swap(input, output);
            continue;
        }
        auto &plan = *plans[i];
        if (!impl->compute(plan, as<ConvObj>(op), context, input, output)) {
            if (i) {
                // The failed layer has not modified its input. Restore that
                // input for the runtime's ordinary NCHW fallback, even when
                // pooled graph buffers have already been reused.
                checkCudnnError(cudnnTransformTensor(
                    handle, &one, plan.xH, input, &zero, plan.xN,
                    op->getInputs(0)->getRawDataPtr<void *>()));
                ++impl->outputTransforms;
            }
            return i;
        }
        ++impl->regionConvSubmissions;
        std::swap(input, output);
    }
    const auto &last = ops[begin + plans.size() - 1];
    checkCudnnError(cudnnTransformTensor(
        handle, &one, lastConv->yH, input, &zero, lastConv->yN,
        last->getOutput()->getRawDataPtr<void *>()));
    ++impl->outputTransforms;
    const char *debug = std::getenv("INFINITENSOR_CONV_DEBUG");
    if (capture == cudaStreamCaptureStatusNone && debug &&
        std::strcmp(debug, "1") == 0)
        std::fprintf(stderr,
                     "[Conv] path=NHWC_REGION ops=%zu convs=%zu "
                     "input_transforms=1 output_transforms=1\n",
                     plans.size(), convs);
    return plans.size();
}
} // namespace infini
