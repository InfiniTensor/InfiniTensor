#pragma once

#include <infini/ops.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace infini::infiniops::dispatch {

using Config = ::infini::ops::Config;
using DataType = ::infini::rt::DataType;
using Handle = ::infini::ops::Handle;
using Tensor = ::infini::ops::Tensor;

// A captured graph owns the exact prepared implementations and workspaces.
// Operator selection and construction happen before entering InfiniRT capture.
using WorkspaceAllocator = std::function<std::shared_ptr<void>(size_t)>;
using PreparedCall =
    std::function<void(const Handle &, const WorkspaceAllocator &)>;
struct CapturePlan {
    struct Entry {
        std::string name;
        size_t implementation;
        PreparedCall invoke;
    };
    std::vector<Entry> operators;
};

class CaptureScope {
    CapturePlan &plan;
    bool replay;
    size_t cursor = 0;
    WorkspaceAllocator workspace;
    static thread_local CaptureScope *active;

  public:
    CaptureScope(CapturePlan &plan, bool replay, WorkspaceAllocator workspace);
    ~CaptureScope();
    CaptureScope(const CaptureScope &) = delete;
    CaptureScope &operator=(const CaptureScope &) = delete;
    static CaptureScope *current() { return active; }
    void call(const std::string &name, size_t implementation,
              const std::function<PreparedCall()> &prepare,
              const Handle &handle);
    void finish() const;
};

void callAdd(const Handle &handle, const Config &config, Tensor input,
             Tensor other, double alpha, Tensor out);
void callMul(const Handle &handle, const Config &config, Tensor input,
             Tensor other, Tensor out);
void callRelu(const Handle &handle, const Config &config, Tensor input,
              Tensor out);
void callRmsNorm(const Handle &handle, const Config &config, Tensor input,
                 Tensor weight, float eps, Tensor out);
void callRotaryEmbedding(const Handle &handle, const Config &config,
                         Tensor input, Tensor posIds, Tensor sinTable,
                         Tensor cosTable, bool isNeox, Tensor out);
void callCat(const Handle &handle, const Config &config,
             std::vector<Tensor> tensors, int64_t dim, Tensor out);
void callMatmul(const Handle &handle, const Config &config, Tensor input,
                Tensor other, Tensor out);
void callConvolution(const Handle &handle, const Config &config, Tensor input,
                     Tensor weight, std::optional<Tensor> bias,
                     std::vector<int64_t> stride, std::vector<int64_t> padding,
                     std::vector<int64_t> dilation, bool transposed,
                     std::vector<int64_t> outputPadding, int64_t groups,
                     Tensor out);
void callNativeBatchNorm(const Handle &handle, const Config &config,
                         Tensor input, std::optional<Tensor> weight,
                         std::optional<Tensor> bias,
                         std::optional<Tensor> runningMean,
                         std::optional<Tensor> runningVar, bool training,
                         double momentum, double eps, Tensor out,
                         Tensor saveMean, Tensor saveInvstd);
void callAvgPool2d(const Handle &handle, const Config &config, Tensor input,
                   std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
                   std::vector<int64_t> padding, bool ceilMode,
                   bool countIncludePad, std::optional<int64_t> divisorOverride,
                   Tensor out);
void callMaxPool2dWithIndices(const Handle &handle, const Config &config,
                              Tensor input, std::vector<int64_t> kernelSize,
                              std::vector<int64_t> stride,
                              std::vector<int64_t> padding,
                              std::vector<int64_t> dilation, bool ceilMode,
                              Tensor out, Tensor indices);
void callMean(const Handle &handle, const Config &config, Tensor input,
              std::optional<std::vector<int64_t>> dim, bool keepdim,
              std::optional<DataType> dtype, Tensor out);
void callGelu(const Handle &handle, const Config &config, Tensor input,
              std::string approximate, Tensor out);
void callHardsigmoid(const Handle &handle, const Config &config, Tensor input,
                     Tensor out);
void callSigmoid(const Handle &handle, const Config &config, Tensor input,
                 Tensor out);
void callSilu(const Handle &handle, const Config &config, Tensor input,
              Tensor out);
void callClip(const Handle &handle, const Config &config, Tensor input,
              std::optional<double> min, std::optional<double> max, Tensor out);
void callSoftmax(const Handle &handle, const Config &config, Tensor input,
                 int64_t dim, std::optional<DataType> dtype, Tensor out);
void callExpandCopy(const Handle &handle, const Config &config, Tensor input,
                    std::vector<int64_t> size, bool implicit, Tensor out);
void callPermuteCopy(const Handle &handle, const Config &config, Tensor input,
                     std::vector<int64_t> dims, Tensor out);
void callSliceCopy(const Handle &handle, const Config &config, Tensor input,
                   int64_t dim, std::optional<int64_t> start,
                   std::optional<int64_t> end, int64_t step, Tensor out);

} // namespace infini::infiniops::dispatch
