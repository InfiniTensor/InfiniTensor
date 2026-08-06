#include "infiniops_dispatch.h"
#include "core/common.h"

#include <base/add.h>
#include <base/avg_pool2d.h>
#include <base/cat.h>
#include <base/clip.h>
#include <base/convolution.h>
#include <base/expand_copy.h>
#include <base/gelu.h>
#include <base/hardsigmoid.h>
#include <base/matmul.h>
#include <base/max_pool2d_with_indices.h>
#include <base/mean.h>
#include <base/mul.h>
#include <base/native_batch_norm.h>
#include <base/permute_copy.h>
#include <base/relu.h>
#include <base/rms_norm.h>
#include <base/rotary_embedding_infinilm.h>
#include <base/sigmoid.h>
#include <base/silu.h>
#include <base/slice_copy.h>
#include <base/softmax.h>

#include <memory>
#include <unordered_map>
#include <utility>

#if defined(__GNUC__) || defined(__clang__)
#define INFINIOPS_OPTIONAL_DISPATCH __attribute__((weak))
#else
#define INFINIOPS_OPTIONAL_DISPATCH
#endif

namespace infini::ops::generated_dispatch {

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

} // namespace infini::ops::generated_dispatch

#undef INFINIOPS_OPTIONAL_DISPATCH

namespace infini::infiniops::dispatch {
namespace {

template <typename Key, bool allowCallFallback = true,
          int fallbackImplementationIndex = -1, typename Factory,
          typename... Args>
void callCached(const Handle &handle, const Config &config, Factory factory,
                const Args &...args) {
    if (factory == nullptr) {
        if constexpr (allowCallFallback) {
            auto fallbackConfig = config;
            if constexpr (fallbackImplementationIndex >= 0) {
                fallbackConfig.set_implementation_index(
                    fallbackImplementationIndex);
            }
            Key::Call(handle, fallbackConfig, args...);
        } else {
            IT_ASSERT(false,
                      "InfiniOps descriptor dispatch is unavailable for this "
                      "operator");
        }
        return;
    }

    using Descriptor = ::infini::ops::Operator<Key>;
    static thread_local std::unordered_map<::infini::ops::detail::CacheKey,
                                           std::unique_ptr<Descriptor>>
        cache;

    auto key = ::infini::ops::CacheKeyBuilder<Key>{}(config, args...);
    auto it = cache.find(key);
    if (it == cache.end()) {
        it = cache.emplace(std::move(key), factory(config, args...)).first;
    }
    (*it->second)(handle, args...);
}

} // namespace

void callAdd(const Handle &handle, const Config &config, Tensor input,
             Tensor other, double alpha, Tensor out) {
#ifdef USE_INFINIOPS_ATEN_KERNELS
    callCached<::infini::ops::Add, true, 1>(
        handle, config, ::infini::ops::generated_dispatch::MakeAdd, input,
        other, alpha, out);
#else
    callCached<::infini::ops::Add>(handle, config,
                                   ::infini::ops::generated_dispatch::MakeAdd,
                                   input, other, alpha, out);
#endif
}

void callMul(const Handle &handle, const Config &config, Tensor input,
             Tensor other, Tensor out) {
#ifdef USE_INFINIOPS_ATEN_KERNELS
    callCached<::infini::ops::Mul, true, 8>(
        handle, config, ::infini::ops::generated_dispatch::MakeMul, input,
        other, out);
#else
    callCached<::infini::ops::Mul>(handle, config,
                                   ::infini::ops::generated_dispatch::MakeMul,
                                   input, other, out);
#endif
}

void callRelu(const Handle &handle, const Config &config, Tensor input,
              Tensor out) {
    callCached<::infini::ops::Relu>(handle, config,
                                    ::infini::ops::generated_dispatch::MakeRelu,
                                    input, out);
}

void callRmsNorm(const Handle &handle, const Config &config, Tensor input,
                 Tensor weight, float eps, Tensor out) {
    callCached<::infini::ops::RmsNorm>(
        handle, config, ::infini::ops::generated_dispatch::MakeRmsNorm, input,
        weight, eps, out);
}

void callRotaryEmbedding(const Handle &handle, const Config &config,
                         Tensor input, Tensor posIds, Tensor sinTable,
                         Tensor cosTable, bool isNeox, Tensor out) {
    callCached<::infini::ops::RotaryEmbeddingInfinilm, false>(
        handle, config,
        ::infini::ops::generated_dispatch::MakeRotaryEmbeddingInfinilm, input,
        posIds, sinTable, cosTable, isNeox, out);
}

void callCat(const Handle &handle, const Config &config,
             std::vector<Tensor> tensors, int64_t dim, Tensor out) {
    callCached<::infini::ops::Cat>(handle, config,
                                   ::infini::ops::generated_dispatch::MakeCat,
                                   tensors, dim, out);
}

void callMatmul(const Handle &handle, const Config &config, Tensor input,
                Tensor other, Tensor out) {
    callCached<::infini::ops::Matmul>(
        handle, config, ::infini::ops::generated_dispatch::MakeMatmul, input,
        other, out);
}

void callConvolution(const Handle &handle, const Config &config, Tensor input,
                     Tensor weight, std::optional<Tensor> bias,
                     std::vector<int64_t> stride, std::vector<int64_t> padding,
                     std::vector<int64_t> dilation, bool transposed,
                     std::vector<int64_t> outputPadding, int64_t groups,
                     Tensor out) {
    callCached<::infini::ops::Convolution>(
        handle, config, ::infini::ops::generated_dispatch::MakeConvolution,
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
    callCached<::infini::ops::NativeBatchNorm>(
        handle, config, ::infini::ops::generated_dispatch::MakeNativeBatchNorm,
        input, weight, bias, runningMean, runningVar, training, momentum, eps,
        out, saveMean, saveInvstd);
}

void callAvgPool2d(const Handle &handle, const Config &config, Tensor input,
                   std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
                   std::vector<int64_t> padding, bool ceilMode,
                   bool countIncludePad, std::optional<int64_t> divisorOverride,
                   Tensor out) {
    callCached<::infini::ops::AvgPool2d>(
        handle, config, ::infini::ops::generated_dispatch::MakeAvgPool2d, input,
        kernelSize, stride, padding, ceilMode, countIncludePad, divisorOverride,
        out);
}

void callMaxPool2dWithIndices(const Handle &handle, const Config &config,
                              Tensor input, std::vector<int64_t> kernelSize,
                              std::vector<int64_t> stride,
                              std::vector<int64_t> padding,
                              std::vector<int64_t> dilation, bool ceilMode,
                              Tensor out, Tensor indices) {
    callCached<::infini::ops::MaxPool2dWithIndices>(
        handle, config,
        ::infini::ops::generated_dispatch::MakeMaxPool2dWithIndices, input,
        kernelSize, stride, padding, dilation, ceilMode, out, indices);
}

void callMean(const Handle &handle, const Config &config, Tensor input,
              std::optional<std::vector<int64_t>> dim, bool keepdim,
              std::optional<DataType> dtype, Tensor out) {
    callCached<::infini::ops::Mean>(handle, config,
                                    ::infini::ops::generated_dispatch::MakeMean,
                                    input, dim, keepdim, dtype, out);
}

void callGelu(const Handle &handle, const Config &config, Tensor input,
              std::string approximate, Tensor out) {
    callCached<::infini::ops::Gelu>(handle, config,
                                    ::infini::ops::generated_dispatch::MakeGelu,
                                    input, approximate, out);
}

void callHardsigmoid(const Handle &handle, const Config &config, Tensor input,
                     Tensor out) {
    callCached<::infini::ops::Hardsigmoid>(
        handle, config, ::infini::ops::generated_dispatch::MakeHardsigmoid,
        input, out);
}

void callSigmoid(const Handle &handle, const Config &config, Tensor input,
                 Tensor out) {
    callCached<::infini::ops::Sigmoid>(
        handle, config, ::infini::ops::generated_dispatch::MakeSigmoid, input,
        out);
}

void callSilu(const Handle &handle, const Config &config, Tensor input,
              Tensor out) {
    callCached<::infini::ops::Silu>(handle, config,
                                    ::infini::ops::generated_dispatch::MakeSilu,
                                    input, out);
}

void callClip(const Handle &handle, const Config &config, Tensor input,
              std::optional<double> min, std::optional<double> max,
              Tensor out) {
    callCached<::infini::ops::Clip>(handle, config,
                                    ::infini::ops::generated_dispatch::MakeClip,
                                    input, min, max, out);
}

void callSoftmax(const Handle &handle, const Config &config, Tensor input,
                 int64_t dim, std::optional<DataType> dtype, Tensor out) {
    callCached<::infini::ops::Softmax>(
        handle, config, ::infini::ops::generated_dispatch::MakeSoftmax, input,
        dim, dtype, out);
}

void callExpandCopy(const Handle &handle, const Config &config, Tensor input,
                    std::vector<int64_t> size, bool implicit, Tensor out) {
    callCached<::infini::ops::ExpandCopy>(
        handle, config, ::infini::ops::generated_dispatch::MakeExpandCopy,
        input, size, implicit, out);
}

void callPermuteCopy(const Handle &handle, const Config &config, Tensor input,
                     std::vector<int64_t> dims, Tensor out) {
    callCached<::infini::ops::PermuteCopy>(
        handle, config, ::infini::ops::generated_dispatch::MakePermuteCopy,
        input, dims, out);
}

void callSliceCopy(const Handle &handle, const Config &config, Tensor input,
                   int64_t dim, std::optional<int64_t> start,
                   std::optional<int64_t> end, int64_t step, Tensor out) {
    callCached<::infini::ops::SliceCopy>(
        handle, config, ::infini::ops::generated_dispatch::MakeSliceCopy, input,
        dim, start, end, step, out);
}

} // namespace infini::infiniops::dispatch
