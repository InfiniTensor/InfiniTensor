#include "infiniops_dispatch.h"
#include "core/common.h"

#include <algorithm>
#if __has_include(<base/add.h>)
#include <base/add.h>
#define INFINITENSOR_HAS_CAPTURE_Add 1
#endif
#if __has_include(<base/avg_pool2d.h>)
#include <base/avg_pool2d.h>
#define INFINITENSOR_HAS_CAPTURE_AvgPool2d 1
#endif
#if __has_include(<base/cat.h>)
#include <base/cat.h>
#define INFINITENSOR_HAS_CAPTURE_Cat 1
#endif
#if __has_include(<base/clip.h>)
#include <base/clip.h>
#define INFINITENSOR_HAS_CAPTURE_Clip 1
#endif
#if __has_include(<base/convolution.h>)
#include <base/convolution.h>
#define INFINITENSOR_HAS_CAPTURE_Convolution 1
#endif
#if __has_include(<base/expand_copy.h>)
#include <base/expand_copy.h>
#define INFINITENSOR_HAS_CAPTURE_ExpandCopy 1
#endif
#if __has_include(<base/gelu.h>)
#include <base/gelu.h>
#define INFINITENSOR_HAS_CAPTURE_Gelu 1
#endif
#if __has_include(<base/hardsigmoid.h>)
#include <base/hardsigmoid.h>
#define INFINITENSOR_HAS_CAPTURE_Hardsigmoid 1
#endif
#if __has_include(<base/matmul.h>)
#include <base/matmul.h>
#define INFINITENSOR_HAS_CAPTURE_Matmul 1
#endif
#if __has_include(<base/max_pool2d_with_indices.h>)
#include <base/max_pool2d_with_indices.h>
#define INFINITENSOR_HAS_CAPTURE_MaxPool2dWithIndices 1
#endif
#if __has_include(<base/mean.h>)
#include <base/mean.h>
#define INFINITENSOR_HAS_CAPTURE_Mean 1
#endif
#if __has_include(<base/mul.h>)
#include <base/mul.h>
#define INFINITENSOR_HAS_CAPTURE_Mul 1
#endif
#if __has_include(<base/native_batch_norm.h>)
#include <base/native_batch_norm.h>
#define INFINITENSOR_HAS_CAPTURE_NativeBatchNorm 1
#endif
#if __has_include(<base/permute_copy.h>)
#include <base/permute_copy.h>
#define INFINITENSOR_HAS_CAPTURE_PermuteCopy 1
#endif
#if __has_include(<base/relu.h>)
#include <base/relu.h>
#define INFINITENSOR_HAS_CAPTURE_Relu 1
#endif
#if __has_include(<base/rms_norm.h>)
#include <base/rms_norm.h>
#define INFINITENSOR_HAS_CAPTURE_RmsNorm 1
#endif
#if __has_include(<base/rotary_embedding_infinilm.h>)
#include <base/rotary_embedding_infinilm.h>
#define INFINITENSOR_HAS_CAPTURE_RotaryEmbeddingInfinilm 1
#endif
#if __has_include(<base/sigmoid.h>)
#include <base/sigmoid.h>
#define INFINITENSOR_HAS_CAPTURE_Sigmoid 1
#endif
#if __has_include(<base/silu.h>)
#include <base/silu.h>
#define INFINITENSOR_HAS_CAPTURE_Silu 1
#endif
#if __has_include(<base/slice_copy.h>)
#include <base/slice_copy.h>
#define INFINITENSOR_HAS_CAPTURE_SliceCopy 1
#endif
#if __has_include(<base/softmax.h>)
#include <base/softmax.h>
#define INFINITENSOR_HAS_CAPTURE_Softmax 1
#endif

#if defined(__GNUC__) || defined(__clang__)
#define INFINIOPS_OPTIONAL_DISPATCH __attribute__((weak))
#else
#define INFINIOPS_OPTIONAL_DISPATCH
#endif

// These operator names are only dispatch keys. Some of their definitions are
// generated only by WITH_TORCH builds of InfiniOps, so the bridge must not
// include their headers when consuming a native-only provider.
namespace infini::ops {
class Add;
class AvgPool2d;
class Cat;
class Clip;
class Convolution;
class ExpandCopy;
class Gelu;
class Hardsigmoid;
class Matmul;
class MaxPool2dWithIndices;
class Mean;
class Mul;
class NativeBatchNorm;
class PermuteCopy;
class Relu;
class RmsNorm;
class RotaryEmbeddingInfinilm;
class Sigmoid;
class Silu;
class SliceCopy;
class Softmax;
} // namespace infini::ops

namespace infini::ops::generated_dispatch {

void CallAdd(const Handle &handle, const Config &config, Tensor input,
             Tensor other, double alpha,
             Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallMul(const Handle &handle, const Config &config, Tensor input,
             Tensor other, Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallRelu(const Handle &handle, const Config &config, Tensor input,
              Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallRmsNorm(const Handle &handle, const Config &config, Tensor input,
                 Tensor weight, float eps,
                 Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallRotaryEmbeddingInfinilm(const Handle &handle, const Config &config,
                                 Tensor input, Tensor posIds, Tensor sinTable,
                                 Tensor cosTable, bool isNeox,
                                 Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallCat(const Handle &handle, const Config &config,
             std::vector<Tensor> tensors, int64_t dim,
             Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallMatmul(const Handle &handle, const Config &config, Tensor input,
                Tensor other, Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallConvolution(const Handle &handle, const Config &config, Tensor input,
                     Tensor weight, std::optional<Tensor> bias,
                     std::vector<int64_t> stride, std::vector<int64_t> padding,
                     std::vector<int64_t> dilation, bool transposed,
                     std::vector<int64_t> outputPadding, int64_t groups,
                     Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallNativeBatchNorm(const Handle &handle, const Config &config,
                         Tensor input, std::optional<Tensor> weight,
                         std::optional<Tensor> bias,
                         std::optional<Tensor> runningMean,
                         std::optional<Tensor> runningVar, bool training,
                         double momentum, double eps, Tensor out,
                         Tensor saveMean,
                         Tensor saveInvstd) INFINIOPS_OPTIONAL_DISPATCH;
void CallAvgPool2d(const Handle &handle, const Config &config, Tensor input,
                   std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
                   std::vector<int64_t> padding, bool ceilMode,
                   bool countIncludePad, std::optional<int64_t> divisorOverride,
                   Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallMaxPool2dWithIndices(const Handle &handle, const Config &config,
                              Tensor input, std::vector<int64_t> kernelSize,
                              std::vector<int64_t> stride,
                              std::vector<int64_t> padding,
                              std::vector<int64_t> dilation, bool ceilMode,
                              Tensor out,
                              Tensor indices) INFINIOPS_OPTIONAL_DISPATCH;
void CallMean(const Handle &handle, const Config &config, Tensor input,
              std::optional<std::vector<int64_t>> dim, bool keepdim,
              std::optional<DataType> dtype,
              Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallGelu(const Handle &handle, const Config &config, Tensor input,
              std::string approximate, Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallHardsigmoid(const Handle &handle, const Config &config, Tensor input,
                     Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallSigmoid(const Handle &handle, const Config &config, Tensor input,
                 Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallSilu(const Handle &handle, const Config &config, Tensor input,
              Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallClip(const Handle &handle, const Config &config, Tensor input,
              std::optional<double> min, std::optional<double> max,
              Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallSoftmax(const Handle &handle, const Config &config, Tensor input,
                 int64_t dim, std::optional<DataType> dtype,
                 Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallExpandCopy(const Handle &handle, const Config &config, Tensor input,
                    std::vector<int64_t> size, bool implicit,
                    Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallPermuteCopy(const Handle &handle, const Config &config, Tensor input,
                     std::vector<int64_t> dims,
                     Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallSliceCopy(const Handle &handle, const Config &config, Tensor input,
                   int64_t dim, std::optional<int64_t> start,
                   std::optional<int64_t> end, int64_t step,
                   Tensor out) INFINIOPS_OPTIONAL_DISPATCH;

std::unique_ptr<Operator<Add>> MakeAdd(const Config &config, Tensor input,
                                       Tensor other, double alpha,
                                       Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
std::unique_ptr<Operator<Mul>> MakeMul(const Config &config, Tensor input,
                                       Tensor other,
                                       Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
std::unique_ptr<Operator<Relu>>
MakeRelu(const Config &config, Tensor input,
         Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
std::unique_ptr<Operator<RmsNorm>>
MakeRmsNorm(const Config &config, Tensor input, Tensor weight, float eps,
            Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
std::unique_ptr<Operator<RotaryEmbeddingInfinilm>>
MakeRotaryEmbeddingInfinilm(const Config &config, Tensor input, Tensor posIds,
                            Tensor sinTable, Tensor cosTable, bool isNeox,
                            Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
std::unique_ptr<Operator<Cat>> MakeCat(const Config &config,
                                       std::vector<Tensor> tensors, int64_t dim,
                                       Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
std::unique_ptr<Operator<Matmul>>
MakeMatmul(const Config &config, Tensor input, Tensor other,
           Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
std::unique_ptr<Operator<Convolution>>
MakeConvolution(const Config &config, Tensor input, Tensor weight,
                std::optional<Tensor> bias, std::vector<int64_t> stride,
                std::vector<int64_t> padding, std::vector<int64_t> dilation,
                bool transposed, std::vector<int64_t> outputPadding,
                int64_t groups, Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
std::unique_ptr<Operator<NativeBatchNorm>>
MakeNativeBatchNorm(const Config &config, Tensor input,
                    std::optional<Tensor> weight, std::optional<Tensor> bias,
                    std::optional<Tensor> runningMean,
                    std::optional<Tensor> runningVar, bool training,
                    double momentum, double eps, Tensor out, Tensor saveMean,
                    Tensor saveInvstd) INFINIOPS_OPTIONAL_DISPATCH;
std::unique_ptr<Operator<AvgPool2d>>
MakeAvgPool2d(const Config &config, Tensor input,
              std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
              std::vector<int64_t> padding, bool ceilMode, bool countIncludePad,
              std::optional<int64_t> divisorOverride,
              Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
std::unique_ptr<Operator<MaxPool2dWithIndices>> MakeMaxPool2dWithIndices(
    const Config &config, Tensor input, std::vector<int64_t> kernelSize,
    std::vector<int64_t> stride, std::vector<int64_t> padding,
    std::vector<int64_t> dilation, bool ceilMode, Tensor out,
    Tensor indices) INFINIOPS_OPTIONAL_DISPATCH;
std::unique_ptr<Operator<Mean>>
MakeMean(const Config &config, Tensor input,
         std::optional<std::vector<int64_t>> dim, bool keepdim,
         std::optional<DataType> dtype, Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
std::unique_ptr<Operator<Gelu>>
MakeGelu(const Config &config, Tensor input, std::string approximate,
         Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
std::unique_ptr<Operator<Hardsigmoid>>
MakeHardsigmoid(const Config &config, Tensor input,
                Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
std::unique_ptr<Operator<Sigmoid>>
MakeSigmoid(const Config &config, Tensor input,
            Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
std::unique_ptr<Operator<Silu>>
MakeSilu(const Config &config, Tensor input,
         Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
std::unique_ptr<Operator<Clip>>
MakeClip(const Config &config, Tensor input, std::optional<double> min,
         std::optional<double> max, Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
std::unique_ptr<Operator<Softmax>>
MakeSoftmax(const Config &config, Tensor input, int64_t dim,
            std::optional<DataType> dtype,
            Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
std::unique_ptr<Operator<ExpandCopy>>
MakeExpandCopy(const Config &config, Tensor input, std::vector<int64_t> size,
               bool implicit, Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
std::unique_ptr<Operator<PermuteCopy>>
MakePermuteCopy(const Config &config, Tensor input, std::vector<int64_t> dims,
                Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
std::unique_ptr<Operator<SliceCopy>>
MakeSliceCopy(const Config &config, Tensor input, int64_t dim,
              std::optional<int64_t> start, std::optional<int64_t> end,
              int64_t step, Tensor out) INFINIOPS_OPTIONAL_DISPATCH;

#define INFINITENSOR_INFINIOPS_DISPATCH_OPERATOR_LIST(X)                       \
    X(Add)                                                                     \
    X(AvgPool2d)                                                               \
    X(Cat)                                                                     \
    X(Clip)                                                                    \
    X(Convolution)                                                             \
    X(ExpandCopy)                                                              \
    X(Gelu)                                                                    \
    X(Hardsigmoid)                                                             \
    X(Matmul)                                                                  \
    X(MaxPool2dWithIndices)                                                    \
    X(Mean)                                                                    \
    X(Mul)                                                                     \
    X(NativeBatchNorm)                                                         \
    X(PermuteCopy)                                                             \
    X(Relu)                                                                    \
    X(RmsNorm)                                                                 \
    X(RotaryEmbeddingInfinilm)                                                 \
    X(Sigmoid)                                                                 \
    X(Silu)                                                                    \
    X(SliceCopy)                                                               \
    X(Softmax)

#define INFINITENSOR_DECLARE_ACTIVE_INDEX_SELECTOR(OperatorName)               \
    std::vector<std::size_t> ActiveImplementationIndicesFor##OperatorName(     \
        Device::Type deviceType) INFINIOPS_OPTIONAL_DISPATCH;

INFINITENSOR_INFINIOPS_DISPATCH_OPERATOR_LIST(
    INFINITENSOR_DECLARE_ACTIVE_INDEX_SELECTOR)

#undef INFINITENSOR_DECLARE_ACTIVE_INDEX_SELECTOR

} // namespace infini::ops::generated_dispatch

#undef INFINIOPS_OPTIONAL_DISPATCH

namespace infini::infiniops::dispatch {
thread_local CaptureScope *CaptureScope::active = nullptr;

CaptureScope::CaptureScope(CapturePlan &plan, bool replay,
                           WorkspaceAllocator workspace)
    : plan(plan), replay(replay), workspace(std::move(workspace)) {
    IT_ASSERT(active == nullptr, "Nested operator capture preparation");
    active = this;
}
CaptureScope::~CaptureScope() { active = nullptr; }
void CaptureScope::finish() const {
    IT_ASSERT(!replay || cursor == plan.operators.size(),
              "Capture operator count differs from preparation");
}
void CaptureScope::call(const std::string &name, size_t implementation,
                        const std::function<PreparedCall()> &prepare,
                        const Handle &handle) {
    if (!replay)
        plan.operators.push_back({name, implementation, prepare()});
    IT_ASSERT(cursor < plan.operators.size(),
              "Unexpected operator during capture");
    const auto &entry = plan.operators[cursor++];
    IT_ASSERT(entry.name == name && entry.implementation == implementation,
              "Capture operator selection differs from preparation");
    entry.invoke(handle, workspace);
}

namespace {

template <typename Key> struct CaptureFactory {
    static const char *name() { return "unavailable operator"; }
    template <typename... Args>
    static PreparedCall prepare(const Config &, const Args &...) {
        IT_ASSERT(false,
                  "InfiniOps build does not expose this prepared operator");
        return {};
    }
};

template <typename Key, typename Factory, typename... Args>
PreparedCall prepareCapture(const char *name, Factory factory,
                            const Config &config, const Args &...args) {
    IT_ASSERT(factory != nullptr, "InfiniOps prepared factory is unavailable");
    std::shared_ptr<::infini::ops::Operator<Key>> op = factory(config, args...);
    IT_ASSERT(
        op != nullptr && op->capture_safe(),
        std::string("InfiniOps implementation is not capture-safe: ") + name +
            " implementation=" + std::to_string(config.implementation_index()));
    return [op, args...](const Handle &handle,
                         const WorkspaceAllocator &allocate) {
        auto invocation = handle.Clone();
        const auto bytes = op->workspace_size_in_bytes();
        auto workspace = bytes ? allocate(bytes) : std::shared_ptr<void>{};
        if (bytes) {
            IT_ASSERT(workspace != nullptr, "Missing capture workspace");
            invocation->set_workspace(workspace.get());
            invocation->set_workspace_size_in_bytes(bytes);
        }
        op->set_handle(*invocation);
        op->set_stream(invocation->stream());
        op->set_workspace(invocation->workspace());
        op->set_workspace_size_in_bytes(invocation->workspace_size_in_bytes());
        static_cast<const Key &>(*op)(args...);
    };
}
#define INFINITENSOR_CAPTURE_FACTORY(Name)                                     \
    template <> struct CaptureFactory<::infini::ops::Name> {                   \
        static const char *name() { return #Name; }                            \
        template <typename... Args>                                            \
        static PreparedCall prepare(const Config &config,                      \
                                    const Args &...args) {                     \
            return prepareCapture<::infini::ops::Name>(                        \
                name(), ::infini::ops::generated_dispatch::Make##Name, config, \
                args...);                                                      \
        }                                                                      \
    };
#ifdef INFINITENSOR_HAS_CAPTURE_Add
INFINITENSOR_CAPTURE_FACTORY(Add)
#endif
#ifdef INFINITENSOR_HAS_CAPTURE_AvgPool2d
INFINITENSOR_CAPTURE_FACTORY(AvgPool2d)
#endif
#ifdef INFINITENSOR_HAS_CAPTURE_Cat
INFINITENSOR_CAPTURE_FACTORY(Cat)
#endif
#ifdef INFINITENSOR_HAS_CAPTURE_Clip
INFINITENSOR_CAPTURE_FACTORY(Clip)
#endif
#ifdef INFINITENSOR_HAS_CAPTURE_Convolution
INFINITENSOR_CAPTURE_FACTORY(Convolution)
#endif
#ifdef INFINITENSOR_HAS_CAPTURE_ExpandCopy
INFINITENSOR_CAPTURE_FACTORY(ExpandCopy)
#endif
#ifdef INFINITENSOR_HAS_CAPTURE_Gelu
INFINITENSOR_CAPTURE_FACTORY(Gelu)
#endif
#ifdef INFINITENSOR_HAS_CAPTURE_Hardsigmoid
INFINITENSOR_CAPTURE_FACTORY(Hardsigmoid)
#endif
#ifdef INFINITENSOR_HAS_CAPTURE_Matmul
INFINITENSOR_CAPTURE_FACTORY(Matmul)
#endif
#ifdef INFINITENSOR_HAS_CAPTURE_MaxPool2dWithIndices
INFINITENSOR_CAPTURE_FACTORY(MaxPool2dWithIndices)
#endif
#ifdef INFINITENSOR_HAS_CAPTURE_Mean
INFINITENSOR_CAPTURE_FACTORY(Mean)
#endif
#ifdef INFINITENSOR_HAS_CAPTURE_Mul
INFINITENSOR_CAPTURE_FACTORY(Mul)
#endif
#ifdef INFINITENSOR_HAS_CAPTURE_NativeBatchNorm
INFINITENSOR_CAPTURE_FACTORY(NativeBatchNorm)
#endif
#ifdef INFINITENSOR_HAS_CAPTURE_PermuteCopy
INFINITENSOR_CAPTURE_FACTORY(PermuteCopy)
#endif
#ifdef INFINITENSOR_HAS_CAPTURE_Relu
INFINITENSOR_CAPTURE_FACTORY(Relu)
#endif
#ifdef INFINITENSOR_HAS_CAPTURE_RmsNorm
INFINITENSOR_CAPTURE_FACTORY(RmsNorm)
#endif
#ifdef INFINITENSOR_HAS_CAPTURE_RotaryEmbeddingInfinilm
INFINITENSOR_CAPTURE_FACTORY(RotaryEmbeddingInfinilm)
#endif
#ifdef INFINITENSOR_HAS_CAPTURE_Sigmoid
INFINITENSOR_CAPTURE_FACTORY(Sigmoid)
#endif
#ifdef INFINITENSOR_HAS_CAPTURE_Silu
INFINITENSOR_CAPTURE_FACTORY(Silu)
#endif
#ifdef INFINITENSOR_HAS_CAPTURE_SliceCopy
INFINITENSOR_CAPTURE_FACTORY(SliceCopy)
#endif
#ifdef INFINITENSOR_HAS_CAPTURE_Softmax
INFINITENSOR_CAPTURE_FACTORY(Softmax)
#endif
#undef INFINITENSOR_CAPTURE_FACTORY

template <typename Key> struct ActiveImplementationQuery;

#define INFINITENSOR_DEFINE_ACTIVE_INDEX_QUERY(OperatorName)                   \
    template <>                                                                \
    struct ActiveImplementationQuery<::infini::ops::OperatorName> {            \
        static std::optional<std::vector<std::size_t>>                         \
        get(::infini::ops::Device::Type deviceType) {                          \
            auto query = ::infini::ops::generated_dispatch::                   \
                ActiveImplementationIndicesFor##OperatorName;                  \
            if (query == nullptr) {                                            \
                return std::nullopt;                                           \
            }                                                                  \
            return query(deviceType);                                          \
        }                                                                      \
    };

INFINITENSOR_INFINIOPS_DISPATCH_OPERATOR_LIST(
    INFINITENSOR_DEFINE_ACTIVE_INDEX_QUERY)

#undef INFINITENSOR_DEFINE_ACTIVE_INDEX_QUERY
#undef INFINITENSOR_INFINIOPS_DISPATCH_OPERATOR_LIST

::infini::ops::Device::Type dispatchDeviceType(const Tensor &tensor) {
    return tensor.device().type();
}

::infini::ops::Device::Type
dispatchDeviceType(const std::vector<Tensor> &tensors) {
    IT_ASSERT(!tensors.empty(), "InfiniOps tensor list must not be empty");
    return tensors.front().device().type();
}

template <typename First, typename... Rest>
::infini::ops::Device::Type firstDispatchDeviceType(const First &first,
                                                    const Rest &...) {
    return dispatchDeviceType(first);
}

template <typename Key, int preferredImplementationIndex = -1, typename Call,
          typename... Args>
void callAvailable(const Handle &handle, const Config &config, Call call,
                   const Args &...args) {
    IT_ASSERT(call != nullptr,
              "InfiniOps does not provide this operator in the current build");

    auto selectedConfig = config;
    const auto activeImplementationIndices =
        ActiveImplementationQuery<Key>::get(firstDispatchDeviceType(args...));
    if (activeImplementationIndices.has_value()) {
        IT_ASSERT(!activeImplementationIndices->empty(),
                  "InfiniOps has no active implementation for this operator "
                  "on the selected device");
        const auto requestedImplementationIndex = config.implementation_index();
        if (std::find(activeImplementationIndices->begin(),
                      activeImplementationIndices->end(),
                      requestedImplementationIndex) ==
            activeImplementationIndices->end()) {
            auto selectedImplementationIndex =
                activeImplementationIndices->front();
            if constexpr (preferredImplementationIndex >= 0) {
                const auto preferred =
                    static_cast<std::size_t>(preferredImplementationIndex);
                if (std::find(activeImplementationIndices->begin(),
                              activeImplementationIndices->end(), preferred) !=
                    activeImplementationIndices->end()) {
                    selectedImplementationIndex = preferred;
                }
            }
            selectedConfig.set_implementation_index(
                selectedImplementationIndex);
        }
    }

    if (auto capture = CaptureScope::current()) {
        capture->call(
            CaptureFactory<Key>::name(), selectedConfig.implementation_index(),
            [&]() {
                return CaptureFactory<Key>::prepare(selectedConfig, args...);
            },
            handle);
        return;
    }
    call(handle, selectedConfig, args...);
}

} // namespace

void callAdd(const Handle &handle, const Config &config, Tensor input,
             Tensor other, double alpha, Tensor out) {
#ifdef USE_INFINIOPS_ATEN_KERNELS
    callAvailable<::infini::ops::Add, 1>(
        handle, config, ::infini::ops::generated_dispatch::CallAdd, input,
        other, alpha, out);
#else
    callAvailable<::infini::ops::Add>(
        handle, config, ::infini::ops::generated_dispatch::CallAdd, input,
        other, alpha, out);
#endif
}

void callMul(const Handle &handle, const Config &config, Tensor input,
             Tensor other, Tensor out) {
#ifdef USE_INFINIOPS_ATEN_KERNELS
    callAvailable<::infini::ops::Mul, 8>(
        handle, config, ::infini::ops::generated_dispatch::CallMul, input,
        other, out);
#else
    callAvailable<::infini::ops::Mul>(
        handle, config, ::infini::ops::generated_dispatch::CallMul, input,
        other, out);
#endif
}

void callRelu(const Handle &handle, const Config &config, Tensor input,
              Tensor out) {
    callAvailable<::infini::ops::Relu>(
        handle, config, ::infini::ops::generated_dispatch::CallRelu, input,
        out);
}

void callRmsNorm(const Handle &handle, const Config &config, Tensor input,
                 Tensor weight, float eps, Tensor out) {
    callAvailable<::infini::ops::RmsNorm>(
        handle, config, ::infini::ops::generated_dispatch::CallRmsNorm, input,
        weight, eps, out);
}

void callRotaryEmbedding(const Handle &handle, const Config &config,
                         Tensor input, Tensor posIds, Tensor sinTable,
                         Tensor cosTable, bool isNeox, Tensor out) {
    callAvailable<::infini::ops::RotaryEmbeddingInfinilm>(
        handle, config,
        ::infini::ops::generated_dispatch::CallRotaryEmbeddingInfinilm, input,
        posIds, sinTable, cosTable, isNeox, out);
}

void callCat(const Handle &handle, const Config &config,
             std::vector<Tensor> tensors, int64_t dim, Tensor out) {
    callAvailable<::infini::ops::Cat>(
        handle, config, ::infini::ops::generated_dispatch::CallCat, tensors,
        dim, out);
}

void callMatmul(const Handle &handle, const Config &config, Tensor input,
                Tensor other, Tensor out) {
    callAvailable<::infini::ops::Matmul>(
        handle, config, ::infini::ops::generated_dispatch::CallMatmul, input,
        other, out);
}

void callConvolution(const Handle &handle, const Config &config, Tensor input,
                     Tensor weight, std::optional<Tensor> bias,
                     std::vector<int64_t> stride, std::vector<int64_t> padding,
                     std::vector<int64_t> dilation, bool transposed,
                     std::vector<int64_t> outputPadding, int64_t groups,
                     Tensor out) {
    callAvailable<::infini::ops::Convolution>(
        handle, config, ::infini::ops::generated_dispatch::CallConvolution,
        input, weight, bias, stride, padding, dilation, transposed,
        outputPadding, groups, out);
}

void callNativeBatchNorm(const Handle &handle, const Config &config,
                         Tensor input, std::optional<Tensor> weight,
                         std::optional<Tensor> bias,
                         std::optional<Tensor> runningMean,
                         std::optional<Tensor> runningVar, bool training,
                         double momentum, double eps, Tensor out,
                         Tensor saveMean, Tensor saveInvstd) {
    callAvailable<::infini::ops::NativeBatchNorm>(
        handle, config, ::infini::ops::generated_dispatch::CallNativeBatchNorm,
        input, weight, bias, runningMean, runningVar, training, momentum, eps,
        out, saveMean, saveInvstd);
}

void callAvgPool2d(const Handle &handle, const Config &config, Tensor input,
                   std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
                   std::vector<int64_t> padding, bool ceilMode,
                   bool countIncludePad, std::optional<int64_t> divisorOverride,
                   Tensor out) {
    callAvailable<::infini::ops::AvgPool2d>(
        handle, config, ::infini::ops::generated_dispatch::CallAvgPool2d, input,
        kernelSize, stride, padding, ceilMode, countIncludePad, divisorOverride,
        out);
}

void callMaxPool2dWithIndices(const Handle &handle, const Config &config,
                              Tensor input, std::vector<int64_t> kernelSize,
                              std::vector<int64_t> stride,
                              std::vector<int64_t> padding,
                              std::vector<int64_t> dilation, bool ceilMode,
                              Tensor out, Tensor indices) {
    callAvailable<::infini::ops::MaxPool2dWithIndices>(
        handle, config,
        ::infini::ops::generated_dispatch::CallMaxPool2dWithIndices, input,
        kernelSize, stride, padding, dilation, ceilMode, out, indices);
}

void callMean(const Handle &handle, const Config &config, Tensor input,
              std::optional<std::vector<int64_t>> dim, bool keepdim,
              std::optional<DataType> dtype, Tensor out) {
    callAvailable<::infini::ops::Mean>(
        handle, config, ::infini::ops::generated_dispatch::CallMean, input, dim,
        keepdim, dtype, out);
}

void callGelu(const Handle &handle, const Config &config, Tensor input,
              std::string approximate, Tensor out) {
    callAvailable<::infini::ops::Gelu>(
        handle, config, ::infini::ops::generated_dispatch::CallGelu, input,
        approximate, out);
}

void callHardsigmoid(const Handle &handle, const Config &config, Tensor input,
                     Tensor out) {
    callAvailable<::infini::ops::Hardsigmoid>(
        handle, config, ::infini::ops::generated_dispatch::CallHardsigmoid,
        input, out);
}

void callSigmoid(const Handle &handle, const Config &config, Tensor input,
                 Tensor out) {
    callAvailable<::infini::ops::Sigmoid>(
        handle, config, ::infini::ops::generated_dispatch::CallSigmoid, input,
        out);
}

void callSilu(const Handle &handle, const Config &config, Tensor input,
              Tensor out) {
    callAvailable<::infini::ops::Silu>(
        handle, config, ::infini::ops::generated_dispatch::CallSilu, input,
        out);
}

void callClip(const Handle &handle, const Config &config, Tensor input,
              std::optional<double> min, std::optional<double> max,
              Tensor out) {
    callAvailable<::infini::ops::Clip>(
        handle, config, ::infini::ops::generated_dispatch::CallClip, input, min,
        max, out);
}

void callSoftmax(const Handle &handle, const Config &config, Tensor input,
                 int64_t dim, std::optional<DataType> dtype, Tensor out) {
    callAvailable<::infini::ops::Softmax>(
        handle, config, ::infini::ops::generated_dispatch::CallSoftmax, input,
        dim, dtype, out);
}

void callExpandCopy(const Handle &handle, const Config &config, Tensor input,
                    std::vector<int64_t> size, bool implicit, Tensor out) {
    callAvailable<::infini::ops::ExpandCopy>(
        handle, config, ::infini::ops::generated_dispatch::CallExpandCopy,
        input, size, implicit, out);
}

void callPermuteCopy(const Handle &handle, const Config &config, Tensor input,
                     std::vector<int64_t> dims, Tensor out) {
    callAvailable<::infini::ops::PermuteCopy>(
        handle, config, ::infini::ops::generated_dispatch::CallPermuteCopy,
        input, dims, out);
}

void callSliceCopy(const Handle &handle, const Config &config, Tensor input,
                   int64_t dim, std::optional<int64_t> start,
                   std::optional<int64_t> end, int64_t step, Tensor out) {
    callAvailable<::infini::ops::SliceCopy>(
        handle, config, ::infini::ops::generated_dispatch::CallSliceCopy, input,
        dim, start, end, step, out);
}

} // namespace infini::infiniops::dispatch
