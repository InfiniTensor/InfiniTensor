#pragma once
#include "core/common.h"
#include "core/graph.h"
#include "core/operator.h"
#include "core/tensor.h"
#include "operators/G2BMM.h"
#include "operators/GBMM.h"
#include "operators/concat.h"
#include "operators/conv.h"
#include "operators/element_wise.h"
#include "operators/extend.h"
#include "operators/gather.h"
#include "operators/matmul.h"
#include "operators/membound.h"
#include "operators/pad.h"
#include "operators/pooling.h"
#include "operators/reduce_mean.h"
#include "operators/reshape.h"
#include "operators/slice.h"
#include "operators/split.h"
#include "operators/unary.h"

namespace infini {

class GraphBuilderObj {
  private:
    Graph g;

  public:
    GraphBuilderObj(Runtime runtime) : g(make_ref<GraphObj>(runtime)) {}

    // tensors
    Tensor tensor(Shape dim, const std::string &dtype);

    // operators
    // conv op
    Operator conv(Tensor input, Tensor weight, Tensor output, int ph, int pw,
                  int sh = 1, int sw = 1, int dh = 1, int dw = 1,
                  Tensor bias = nullptr);
    Operator conv(Tensor input, Tensor weight, int ph, int pw, int sh = 1,
                  int sw = 1, int dh = 1, int dw = 1, Tensor bias = nullptr);
    Operator conv(Tensor input, Tensor weight, Tensor output,
                  ConvBaseObj::PaddingMode pm, int sh = 1, int sw = 1,
                  int dh = 1, int dw = 1, Tensor bias = nullptr);
    Operator conv(Tensor input, Tensor weight, ConvBaseObj::PaddingMode pm,
                  int sh = 1, int sw = 1, int dh = 1, int dw = 1,
                  Tensor bias = nullptr);
    // matmul op
    Operator matmul(Tensor A, Tensor B, Tensor C, bool transA = false,
                    bool transB = false, Tensor bias = nullptr,
                    ActType act = ActType::None);
    Operator matmul(Tensor A, Tensor B, bool transA = false,
                    bool transB = false, Tensor bias = nullptr,
                    ActType act = ActType::None);
    // conv trans op
    Operator convTrans(Tensor input, Tensor weight, Tensor output, int ph,
                       int pw, int sh = 1, int sw = 1, int dh = 1, int dw = 1,
                       int oph = 0, int opw = 0, int group = 1,
                       Tensor bias = nullptr, ActType act = ActType::None);
    Operator convTrans(Tensor input, Tensor weight, int ph, int pw, int sh = 1,
                       int sw = 1, int dh = 1, int dw = 1, int oph = 0,
                       int opw = 0, int group = 1, Tensor bias = nullptr,
                       ActType act = ActType::None);
    Operator convTrans(Tensor input, Tensor weight, Tensor output,
                       ConvBaseObj::PaddingMode pm, int sh = 1, int sw = 1,
                       int dh = 1, int dw = 1, int oph = 0, int opw = 0,
                       int group = 1, Tensor bias = nullptr,
                       ActType act = ActType::None);
    Operator convTrans(Tensor input, Tensor weight, ConvBaseObj::PaddingMode pm,
                       int sh = 1, int sw = 1, int dh = 1, int dw = 1,
                       int oph = 0, int opw = 0, int group = 1,
                       Tensor bias = nullptr, ActType act = ActType::None);
    // g2bmm op
    Operator g2bmm(Tensor A, Tensor B, Tensor C, const int width,
                   const int dilation, Tensor bias = nullptr,
                   ActType act = ActType::None);
    Operator g2bmm(Tensor A, Tensor B, const int width, const int dilation,
                   Tensor bias = nullptr, ActType act = ActType::None);
    // gbmm-like op
    Operator gbmml(Tensor A, Tensor B, Tensor C, const int dilation,
                   Tensor bias = nullptr, ActType act = ActType::None);
    Operator gbmml(Tensor A, Tensor B, const int dilation,
                   Tensor bias = nullptr, ActType act = ActType::None);
    // pad op
    Operator pad(Tensor input, Tensor output, const vector<int> &pads,
                 const optional<const vector<int>> &axis);
    Operator pad(Tensor input, const vector<int> &pads,
                 const optional<const vector<int>> &axis);
    // slice op
    Operator slice(Tensor input, Tensor output, const vector<int> &starts,
                   const vector<int> &ends,
                   const optional<const vector<int>> &axis,
                   const optional<const vector<int>> &steps);
    Operator slice(Tensor input, const vector<int> &starts,
                   const vector<int> &ends,
                   const optional<const vector<int>> &axis,
                   const optional<const vector<int>> &steps);
    // concat op
    Operator concat(TensorVec inputs, Tensor output, int dim);
    Operator concat(TensorVec inputs, int dim);
    // split op
    Operator split(Tensor input, std::optional<TensorVec> outputs, int dim,
                   int num);
    Operator split(Tensor input, int dim, int num);
    Operator split(Tensor input, std::optional<TensorVec> outputs, int dim,
                   const vector<int> &ratio);
    Operator split(Tensor input, int dim, const vector<int> &ratio);
    // transpose op
    // TODO
    // extend op
    Operator extend(Tensor input, Tensor output, int dim, int num);
    Operator extend(Tensor input, int dim, int num);
    // max pool op
    Operator maxpool(Tensor input, Tensor output, int kh, int kw, int dh,
                     int dw, int ph, int pw, int sh, int sw);
    Operator maxpool(Tensor input, int kh, int kw, int dh, int dw, int ph,
                     int pw, int sh, int sw);
    // average pool op
    Operator avgpool(Tensor input, Tensor output, int kh, int kw, int dh,
                     int dw, int ph, int pw, int sh, int sw);
    Operator avgpool(Tensor input, int kh, int kw, int dh, int dw, int ph,
                     int pw, int sh, int sw);
    // element wise op
    Operator add(Tensor input0, Tensor input1, Tensor output);
    Operator add(Tensor input0, Tensor input1);
    Operator sub(Tensor input0, Tensor input1, Tensor output);
    Operator sub(Tensor input0, Tensor input1);
    Operator mul(Tensor input0, Tensor input1, Tensor output);
    Operator mul(Tensor input0, Tensor input1);
    Operator div(Tensor input0, Tensor input1, Tensor output);
    Operator div(Tensor input0, Tensor input1);
    Operator pow(Tensor input0, Tensor input1, Tensor output);
    Operator pow(Tensor input0, Tensor input1);
    // gather op
    Operator gather(Tensor input, Tensor index, Tensor output, int axis);
    Operator gather(Tensor input, Tensor index, int axis);
    // reduce mean op
    // TODO
    // reshape op
    Operator reshape(Tensor input, Tensor output, const Shape &dims);
    Operator reshape(Tensor input, const Shape &dims);
    Operator flatten(Tensor input, Tensor output);
    Operator flatten(Tensor input);
    Operator identity(Tensor input, Tensor output);
    Operator identity(Tensor input);
    // unary op
    // TODO: batch norm
    Operator softmax(Tensor input, Tensor output);
    Operator softmax(Tensor input);
    // TODO: activation
    Operator relu(Tensor input, Tensor output);
    Operator relu(Tensor input);
    Operator sigmoid(Tensor input, Tensor output);
    Operator sigmoid(Tensor input);
    Operator tanh(Tensor input, Tensor output);
    Operator tanh(Tensor input);
    Operator abs(Tensor input, Tensor output);
    Operator abs(Tensor input);
    Operator reduceMean(Tensor input, Tensor Output, int axis);
    // resize op
    // TODO
    // membound op
    Operator memBound(const TensorVec &inputs, const TensorVec &outputs,
                      const std::vector<nnet::Tensor> &nnetInputs,
                      nnet::Expr expr, double exec_time, std::string hint = {});
};

} // namespace infini
