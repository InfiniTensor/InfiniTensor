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
class Div;
class GridSample;
class RoiAlign;
class ScatterNd;
class MaskedScatter;
class Scatter;
class Tile;
class Eq;
class Ge;
class Pow;
class Maximum;
class Sum;
class Amax;
class Tanh;
class Abs;
class Sign;
class Rsqrt;
class Log;
class Resize2d;
class InstanceNorm;
class Less;
class Sin;
class Cos;
class Erf;
class Neg;
class LogicalNot;
class Sqrt;
class Argmax;
class AvgPool2d;
class Cat;
class Cast;
class Clip;
class Convolution;
class Copy;
class ExpandCopy;
class Gelu;
class Hardsigmoid;
class IndexSelect;
class GatherAxis;
class LayerNorm;
class Matmul;
class MaxPool2dWithIndices;
class MaxPool2d;
class Mean;
class Mul;
class NativeBatchNorm;
class Norm;
class PermuteCopy;
class Relu;
class RmsNorm;
class RotaryEmbeddingInfinilm;
class Sigmoid;
class Silu;
class SliceCopy;
class Softmax;
class Where;
} // namespace infini::ops

namespace infini::ops::generated_dispatch {

void CallCopy(const Handle &handle, const Config &config, Tensor input,
              bool nonBlocking, Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallCast(const Handle &handle, const Config &config, Tensor input,
              Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallIndexSelect(const Handle &handle, const Config &config, Tensor input,
                     Tensor indices, int64_t axis, Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallGatherAxis(const Handle &handle, const Config &config, Tensor input,
                    Tensor indices, int64_t axis, Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallArgmax(const Handle &handle, const Config &config, Tensor input,
                std::optional<int64_t> axis, bool keepdims, Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallNorm(const Handle &handle, const Config &config, Tensor input,
              std::optional<double> p, std::vector<int64_t> axes, bool keepdims,
              Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallLayerNorm(const Handle &handle, const Config &config, Tensor input,
                   Tensor scale, std::optional<Tensor> bias,
                   float eps, Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallWhere(const Handle &handle, const Config &config, Tensor condition,
               Tensor input, Tensor other, Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallAdd(const Handle &handle, const Config &config, Tensor input,
             Tensor other, double alpha,
             Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallMul(const Handle &handle, const Config &config, Tensor input,
             Tensor other, Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallDiv(const Handle &handle, const Config &config, Tensor input,
             Tensor other, Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallGridSample(const Handle &handle, const Config &config, Tensor input,
                    Tensor grid, bool nearest, bool reflection,
                    bool alignCorners, Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallRoiAlign(const Handle &handle, const Config &config, Tensor input,
                  Tensor boxes, int64_t samplingRatio, float spatialScale,
                  bool aligned, Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallScatterNd(const Handle &handle, const Config &config, Tensor input,
                    Tensor indices, Tensor updates, bool add,
                    Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallMaskedScatter(const Handle &handle, const Config &config, Tensor input,
                        Tensor mask, Tensor source, Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallScatter(const Handle &handle, const Config &config, Tensor input, Tensor index,
                 Tensor source, int64_t axis, std::string reduction,
                 Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallRelu(const Handle &handle, const Config &config, Tensor input,
              Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallTile(const Handle &handle, const Config &config, Tensor input,
              Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallEq(const Handle &handle, const Config &config, Tensor input,
            Tensor other, Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallGe(const Handle &handle, const Config &config, Tensor input,
            Tensor other, Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallPow(const Handle &handle, const Config &config, Tensor input,
             Tensor exponent, Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallMaximum(const Handle &handle, const Config &config, Tensor input,
                 Tensor other, Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallLess(const Handle &handle, const Config &config, Tensor input,
            Tensor other, Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallSin(const Handle &handle, const Config &config, Tensor input,
             Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallCos(const Handle &handle, const Config &config, Tensor input,
             Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallErf(const Handle &handle, const Config &config, Tensor input,
             Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallNeg(const Handle &handle, const Config &config, Tensor input,
             Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallLogicalNot(const Handle &handle, const Config &config, Tensor input,
                    Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallSqrt(const Handle &handle, const Config &config, Tensor input,
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
void CallMaxPool2d(const Handle &handle, const Config &config, Tensor input,
                   std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
                   std::vector<int64_t> padding, std::vector<int64_t> dilation,
                   bool ceilMode, Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallMean(const Handle &handle, const Config &config, Tensor input,
              std::optional<std::vector<int64_t>> dim, bool keepdim,
              std::optional<DataType> dtype,
              Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallSum(const Handle &handle, const Config &config, Tensor input,
             std::optional<std::vector<int64_t>> dim, bool keepdim,
             std::optional<DataType> dtype, Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallAmax(const Handle &handle, const Config &config, Tensor input,
              std::vector<int64_t> dim, bool keepdim, Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallTanh(const Handle &handle, const Config &config, Tensor input,
              Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallAbs(const Handle &handle, const Config &config, Tensor input,
              Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallSign(const Handle &handle, const Config &config, Tensor input,
              Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallRsqrt(const Handle &handle, const Config &config, Tensor input,
              Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallLog(const Handle &handle, const Config &config, Tensor input,
              Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallResize2d(const Handle &handle, const Config &config, Tensor input,
                  int64_t mode, int64_t coordinates, int64_t rounding,
                  float heightScale, float widthScale,
                  Tensor out) INFINIOPS_OPTIONAL_DISPATCH;
void CallInstanceNorm(const Handle &handle, const Config &config, Tensor input,
                       Tensor scale, Tensor bias, float epsilon,
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
    X(Div)                                                                     \
    X(GridSample)                                                              \
    X(RoiAlign)                                                                \
    X(ScatterNd)                                                               \
    X(MaskedScatter)                                                           \
    X(Scatter)                                                                 \
    X(Tile)                                                                    \
    X(Eq)                                                                      \
    X(Ge)                                                                      \
    X(Pow)                                                                     \
    X(Maximum)                                                                 \
    X(Less)                                                                    \
    X(Sin)                                                                 \
    X(Cos)                                                                 \
    X(Erf)                                                                 \
    X(Neg)                                                                 \
    X(LogicalNot)                                                              \
    X(Sqrt)                                                                \
    X(Argmax)                                                                  \
    X(AvgPool2d)                                                               \
    X(Cat)                                                                     \
    X(Cast)                                                                    \
    X(Clip)                                                                    \
    X(Convolution)                                                             \
    X(Copy)                                                                    \
    X(ExpandCopy)                                                              \
    X(Gelu)                                                                    \
    X(Hardsigmoid)                                                             \
    X(IndexSelect)                                                             \
    X(GatherAxis)                                                              \
    X(LayerNorm)                                                               \
    X(Matmul)                                                                  \
    X(MaxPool2dWithIndices)                                                    \
    X(MaxPool2d)                                                               \
    X(Mean)                                                                    \
    X(Sum)                                                                     \
    X(Amax)                                                                    \
    X(Tanh)                                                                    \
    X(Abs)                                                                  \
    X(Sign)                                                                 \
    X(Rsqrt)                                                                \
    X(Log)                                                                  \
    X(Resize2d)                                                                \
    X(InstanceNorm)                                                            \
    X(Mul)                                                                     \
    X(NativeBatchNorm)                                                         \
    X(Norm)                                                                    \
    X(PermuteCopy)                                                             \
    X(Relu)                                                                    \
    X(RmsNorm)                                                                 \
    X(RotaryEmbeddingInfinilm)                                                 \
    X(Sigmoid)                                                                 \
    X(Silu)                                                                    \
    X(SliceCopy)                                                               \
    X(Softmax)                                                                 \
    X(Where)

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

void callCopy(const Handle &handle, const Config &config, Tensor input,
              bool nonBlocking, Tensor out) {
    callAvailable<::infini::ops::Copy>(
        handle, config, ::infini::ops::generated_dispatch::CallCopy, input,
        nonBlocking, out);
}

void callWhere(const Handle &handle, const Config &config, Tensor condition,
               Tensor input, Tensor other, Tensor out) {
    callAvailable<::infini::ops::Where>(
        handle, config, ::infini::ops::generated_dispatch::CallWhere,
        condition, input, other, out);
}

void callCast(const Handle &handle, const Config &config, Tensor input,
              Tensor out) {
    callAvailable<::infini::ops::Cast>(
        handle, config, ::infini::ops::generated_dispatch::CallCast, input, out);
}

void callIndexSelect(const Handle &handle, const Config &config, Tensor input,
                     Tensor indices, int64_t axis, Tensor out) {
    callAvailable<::infini::ops::IndexSelect>(
        handle, config, ::infini::ops::generated_dispatch::CallIndexSelect,
        input, indices, axis, out);
}

void callGatherAxis(const Handle &handle, const Config &config, Tensor input,
                    Tensor indices, int64_t axis, Tensor out) {
    callAvailable<::infini::ops::GatherAxis>(
        handle, config, ::infini::ops::generated_dispatch::CallGatherAxis,
        input, indices, axis, out);
}

void callArgmax(const Handle &handle, const Config &config, Tensor input,
                std::optional<int64_t> axis, bool keepdims, Tensor out) {
    callAvailable<::infini::ops::Argmax>(
        handle, config, ::infini::ops::generated_dispatch::CallArgmax,
        input, axis, keepdims, out);
}

void callNorm(const Handle &handle, const Config &config, Tensor input,
              std::optional<double> p, std::vector<int64_t> axes, bool keepdims,
              Tensor out) {
    callAvailable<::infini::ops::Norm>(
        handle, config, ::infini::ops::generated_dispatch::CallNorm,
        input, p, axes, keepdims, out);
}

void callLayerNorm(const Handle &handle, const Config &config, Tensor input,
                   Tensor scale, std::optional<Tensor> bias,
                   float eps, Tensor out) {
    callAvailable<::infini::ops::LayerNorm>(
        handle, config, ::infini::ops::generated_dispatch::CallLayerNorm,
        input, scale, bias, eps, out);
}

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

void callDiv(const Handle &handle, const Config &config, Tensor input,
             Tensor other, Tensor out) {
    using Call = void (*)(const Handle &, const Config &, Tensor, Tensor, Tensor);
    callAvailable<::infini::ops::Div>(
        handle, config,
        static_cast<Call>(::infini::ops::generated_dispatch::CallDiv), input,
        other, out);
}

void callGridSample(const Handle &handle, const Config &config, Tensor input,
                    Tensor grid, bool nearest, bool reflection,
                    bool alignCorners, Tensor out) {
    callAvailable<::infini::ops::GridSample>(
        handle, config, ::infini::ops::generated_dispatch::CallGridSample,
        input, grid, nearest, reflection, alignCorners, out);
}

void callSin(const Handle &handle, const Config &config, Tensor input, Tensor out) {
    callAvailable<::infini::ops::Sin>(
        handle, config, ::infini::ops::generated_dispatch::CallSin, input, out);
}

void callCos(const Handle &handle, const Config &config, Tensor input, Tensor out) {
    callAvailable<::infini::ops::Cos>(
        handle, config, ::infini::ops::generated_dispatch::CallCos, input, out);
}

void callErf(const Handle &handle, const Config &config, Tensor input, Tensor out) {
    callAvailable<::infini::ops::Erf>(
        handle, config, ::infini::ops::generated_dispatch::CallErf, input, out);
}

void callNeg(const Handle &handle, const Config &config, Tensor input, Tensor out) {
    callAvailable<::infini::ops::Neg>(
        handle, config, ::infini::ops::generated_dispatch::CallNeg, input, out);
}

void callLogicalNot(const Handle &handle, const Config &config, Tensor input, Tensor out) {
    callAvailable<::infini::ops::LogicalNot>(
        handle, config, ::infini::ops::generated_dispatch::CallLogicalNot, input, out);
}

void callSqrt(const Handle &handle, const Config &config, Tensor input, Tensor out) {
    callAvailable<::infini::ops::Sqrt>(
        handle, config, ::infini::ops::generated_dispatch::CallSqrt, input, out);
}

void callRoiAlign(const Handle &handle, const Config &config, Tensor input,
                  Tensor boxes, int64_t samplingRatio, float spatialScale,
                  bool aligned, Tensor out) {
    callAvailable<::infini::ops::RoiAlign>(
        handle, config, ::infini::ops::generated_dispatch::CallRoiAlign,
        input, boxes, samplingRatio, spatialScale, aligned, out);
}

void callScatterNd(const Handle &handle, const Config &config, Tensor input,
                    Tensor indices, Tensor updates, bool add, Tensor out) {
    callAvailable<::infini::ops::ScatterNd>(
        handle, config, ::infini::ops::generated_dispatch::CallScatterNd,
        input, indices, updates, add, out);
}

void callTile(const Handle &handle, const Config &config, Tensor input, Tensor out) {
    callAvailable<::infini::ops::Tile>(
        handle, config, ::infini::ops::generated_dispatch::CallTile, input, out);
}

void callEq(const Handle &handle, const Config &config, Tensor input, Tensor other, Tensor out) {
    using Call = void (*)(const Handle &, const Config &, Tensor, Tensor, Tensor);
    callAvailable<::infini::ops::Eq>(
        handle, config, static_cast<Call>(::infini::ops::generated_dispatch::CallEq),
        input, other, out);
}

void callGe(const Handle &handle, const Config &config, Tensor input, Tensor other, Tensor out) {
    using Call = void (*)(const Handle &, const Config &, Tensor, Tensor, Tensor);
    callAvailable<::infini::ops::Ge>(
        handle, config, static_cast<Call>(::infini::ops::generated_dispatch::CallGe),
        input, other, out);
}

void callPow(const Handle &handle, const Config &config, Tensor input, Tensor exponent, Tensor out) {
    using Call = void (*)(const Handle &, const Config &, Tensor, Tensor, Tensor);
    callAvailable<::infini::ops::Pow>(
        handle, config, static_cast<Call>(::infini::ops::generated_dispatch::CallPow),
        input, exponent, out);
}

void callMaximum(const Handle &handle, const Config &config, Tensor input, Tensor other, Tensor out) {
    callAvailable<::infini::ops::Maximum>(
        handle, config, ::infini::ops::generated_dispatch::CallMaximum, input, other, out);
}

void callLess(const Handle &handle, const Config &config, Tensor input, Tensor other, Tensor out) {
    using Call = void (*)(const Handle &, const Config &, Tensor, Tensor, Tensor);
    callAvailable<::infini::ops::Less>(
        handle, config, static_cast<Call>(::infini::ops::generated_dispatch::CallLess),
        input, other, out);
}

void callScatter(const Handle &handle, const Config &config, Tensor input, Tensor index,
                 Tensor source, int64_t axis, std::string reduction, Tensor out) {
    using Call = void (*)(const Handle &, const Config &, Tensor, Tensor, Tensor,
                          int64_t, std::string, Tensor);
    callAvailable<::infini::ops::Scatter>(
        handle, config, static_cast<Call>(::infini::ops::generated_dispatch::CallScatter),
        input, index, source, axis, reduction, out);
}

void callMaskedScatter(const Handle &handle, const Config &config, Tensor input,
                        Tensor mask, Tensor source, Tensor out) {
    callAvailable<::infini::ops::MaskedScatter>(
        handle, config, ::infini::ops::generated_dispatch::CallMaskedScatter,
        input, mask, source, out);
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

void callMaxPool2d(const Handle &handle, const Config &config, Tensor input,
                   int64_t kh, int64_t kw, int64_t sh, int64_t sw, int64_t ph,
                   int64_t pw, bool ceilMode, Tensor out) {
    callAvailable<::infini::ops::MaxPool2d>(
        handle, config, ::infini::ops::generated_dispatch::CallMaxPool2d,
        input, std::vector<int64_t>{kh, kw}, std::vector<int64_t>{sh, sw},
        std::vector<int64_t>{ph, pw}, std::vector<int64_t>{1, 1}, ceilMode, out);
}

void callMean(const Handle &handle, const Config &config, Tensor input,
              std::optional<std::vector<int64_t>> dim, bool keepdim,
              std::optional<DataType> dtype, Tensor out) {
    callAvailable<::infini::ops::Mean>(
        handle, config, ::infini::ops::generated_dispatch::CallMean, input, dim,
        keepdim, dtype, out);
}

void callSum(const Handle &handle, const Config &config, Tensor input,
             std::optional<std::vector<int64_t>> dim, bool keepdim,
             std::optional<DataType> dtype, Tensor out) {
    callAvailable<::infini::ops::Sum>(
        handle, config, ::infini::ops::generated_dispatch::CallSum,
        input, dim, keepdim, dtype, out);
}

void callAmax(const Handle &handle, const Config &config, Tensor input,
              std::vector<int64_t> dim, bool keepdim, Tensor out) {
    callAvailable<::infini::ops::Amax>(
        handle, config, ::infini::ops::generated_dispatch::CallAmax,
        input, dim, keepdim, out);
}

void callTanh(const Handle &handle, const Config &config, Tensor input, Tensor out) {
    callAvailable<::infini::ops::Tanh>(
        handle, config, ::infini::ops::generated_dispatch::CallTanh, input, out);
}

void callAbs(const Handle &handle, const Config &config, Tensor input, Tensor out) {
    callAvailable<::infini::ops::Abs>(
        handle, config, ::infini::ops::generated_dispatch::CallAbs, input, out);
}

void callSign(const Handle &handle, const Config &config, Tensor input, Tensor out) {
    callAvailable<::infini::ops::Sign>(
        handle, config, ::infini::ops::generated_dispatch::CallSign, input, out);
}

void callRsqrt(const Handle &handle, const Config &config, Tensor input, Tensor out) {
    callAvailable<::infini::ops::Rsqrt>(
        handle, config, ::infini::ops::generated_dispatch::CallRsqrt, input, out);
}

void callLog(const Handle &handle, const Config &config, Tensor input, Tensor out) {
    callAvailable<::infini::ops::Log>(
        handle, config, ::infini::ops::generated_dispatch::CallLog, input, out);
}

void callResize2d(const Handle &handle, const Config &config, Tensor input,
                  int64_t mode, int64_t coordinates, int64_t rounding,
                  float heightScale, float widthScale, Tensor out) {
    callAvailable<::infini::ops::Resize2d>(
        handle, config, ::infini::ops::generated_dispatch::CallResize2d,
        input, mode, coordinates, rounding, heightScale, widthScale, out);
}

void callInstanceNorm(const Handle &handle, const Config &config, Tensor input,
                       Tensor scale, Tensor bias, float epsilon, Tensor out) {
    callAvailable<::infini::ops::InstanceNorm>(
        handle, config, ::infini::ops::generated_dispatch::CallInstanceNorm,
        input, scale, bias, epsilon, out);
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
