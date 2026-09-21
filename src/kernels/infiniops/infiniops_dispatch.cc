#include "infiniops_dispatch.h"
#include "core/common.h"

#include <algorithm>

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
namespace {

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
