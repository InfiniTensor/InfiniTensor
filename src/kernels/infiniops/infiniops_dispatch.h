#pragma once

#include <infini/ops.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace infini::infiniops::dispatch {

using Config = ::infini::ops::Config;
using DataType = ::infini::rt::DataType;
using Handle = ::infini::ops::Handle;
using Tensor = ::infini::ops::Tensor;

void callCopy(const Handle &handle, const Config &config, Tensor input,
              bool nonBlocking, Tensor out);
void callCast(const Handle &handle, const Config &config, Tensor input,
              Tensor out);
void callIndexSelect(const Handle &handle, const Config &config, Tensor input,
                     Tensor indices, int64_t axis, Tensor out);
void callGatherAxis(const Handle &handle, const Config &config, Tensor input,
                    Tensor indices, int64_t axis, Tensor out);
void callArgmax(const Handle &handle, const Config &config, Tensor input,
                std::optional<int64_t> axis, bool keepdims, Tensor out);
void callNorm(const Handle &handle, const Config &config, Tensor input,
              std::optional<double> p, std::vector<int64_t> axes, bool keepdims,
              Tensor out);
void callLayerNorm(const Handle &handle, const Config &config, Tensor input,
                   Tensor scale, std::optional<Tensor> bias,
                   float eps, Tensor out);
void callWhere(const Handle &handle, const Config &config, Tensor condition,
               Tensor input, Tensor other, Tensor out);
void callAdd(const Handle &handle, const Config &config, Tensor input,
             Tensor other, double alpha, Tensor out);
void callMul(const Handle &handle, const Config &config, Tensor input,
             Tensor other, Tensor out);
void callDiv(const Handle &handle, const Config &config, Tensor input,
             Tensor other, Tensor out);
void callTile(const Handle &handle, const Config &config, Tensor input, Tensor out);
void callEq(const Handle &handle, const Config &config, Tensor input, Tensor other, Tensor out);
void callGe(const Handle &handle, const Config &config, Tensor input, Tensor other, Tensor out);
void callPow(const Handle &handle, const Config &config, Tensor input, Tensor exponent, Tensor out);
void callMaximum(const Handle &handle, const Config &config, Tensor input, Tensor other, Tensor out);
void callLess(const Handle &handle, const Config &config, Tensor input, Tensor other, Tensor out);
void callSin(const Handle &handle, const Config &config, Tensor input, Tensor out);
void callCos(const Handle &handle, const Config &config, Tensor input, Tensor out);
void callErf(const Handle &handle, const Config &config, Tensor input, Tensor out);
void callNeg(const Handle &handle, const Config &config, Tensor input, Tensor out);
void callLogicalNot(const Handle &handle, const Config &config, Tensor input, Tensor out);
void callSqrt(const Handle &handle, const Config &config, Tensor input, Tensor out);
void callGridSample(const Handle &handle, const Config &config, Tensor input,
                    Tensor grid, bool nearest, bool reflection,
                    bool alignCorners, Tensor out);
void callRoiAlign(const Handle &handle, const Config &config, Tensor input,
                  Tensor boxes, int64_t samplingRatio, float spatialScale,
                  bool aligned, Tensor out);
void callScatterNd(const Handle &handle, const Config &config, Tensor input,
                    Tensor indices, Tensor updates, bool add, Tensor out);
void callMaskedScatter(const Handle &handle, const Config &config, Tensor input,
                        Tensor mask, Tensor source, Tensor out);
void callScatter(const Handle &handle, const Config &config, Tensor input, Tensor index,
                 Tensor source, int64_t axis, std::string reduction, Tensor out);
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
void callMaxPool2d(const Handle &handle, const Config &config, Tensor input,
                   int64_t kh, int64_t kw, int64_t sh, int64_t sw, int64_t ph,
                   int64_t pw, bool ceilMode, Tensor out);
void callMean(const Handle &handle, const Config &config, Tensor input,
              std::optional<std::vector<int64_t>> dim, bool keepdim,
              std::optional<DataType> dtype, Tensor out);
void callSum(const Handle &handle, const Config &config, Tensor input,
             std::optional<std::vector<int64_t>> dim, bool keepdim,
             std::optional<DataType> dtype, Tensor out);
void callAmax(const Handle &handle, const Config &config, Tensor input,
              std::vector<int64_t> dim, bool keepdim, Tensor out);
void callTanh(const Handle &handle, const Config &config, Tensor input, Tensor out);
void callAbs(const Handle &handle, const Config &config, Tensor input, Tensor out);
void callSign(const Handle &handle, const Config &config, Tensor input, Tensor out);
void callRsqrt(const Handle &handle, const Config &config, Tensor input, Tensor out);
void callLog(const Handle &handle, const Config &config, Tensor input, Tensor out);
void callResize2d(const Handle &handle, const Config &config, Tensor input,
                  int64_t mode, int64_t coordinates, int64_t rounding,
                  float heightScale, float widthScale, Tensor out);
void callInstanceNorm(const Handle &handle, const Config &config, Tensor input,
                       Tensor scale, Tensor bias, float epsilon, Tensor out);
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
