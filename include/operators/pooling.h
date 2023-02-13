#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief The base class for AvgPool and MaxPool.
 *
 */
class PoolingObj : public OperatorObj {
  private:
    int kh, kw;
    int dh, dw;
    int ph, pw;
    int sh, sw;
    int n, c, h, w;

  public:
    /**
     * @brief Construct a new Pooling object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param optype Operator type of this pooling operator.
     * @param input The input tensor.
     * @param output The output tensor.
     * @param kh Kernel height.
     * @param kw Kernel width.
     * FIXME: Dilated pooling is not supported for many frameworks?
     * @param dh Dilation at the height dimension.
     * @param dw Dilation at the width dimension.
     * FIXME: Auto padding using padding mode.
     * @param ph Padding at the height dimension.
     * @param pw Padding at the width dimension.
     * @param sh Stride at the height dimension.
     * @param sw Stride at the width dimension.
     */
    PoolingObj(GraphObj *graph, OpType optype, Tensor input, Tensor output,
               int kh, int kw, int dh, int dw, int ph, int pw, int sh, int sw);
    OP_CLONE(PoolingObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;
    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

    int getKh() const { return kh; }
    int getKw() const { return kw; }
    int getDh() const { return dh; }
    int getDw() const { return dw; }
    int getPh() const { return ph; }
    int getPw() const { return pw; }
    int getSh() const { return sh; }
    int getSw() const { return sw; }

    auto getPadStrideDilation() const { return tuple(ph, pw, sh, sw, dh, dw); }
    auto getNCHWRS() const { return tuple(n, c, h, w, kh, kw); }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

class MaxPoolObj : public PoolingObj {
  public:
    MaxPoolObj(GraphObj *graph, Tensor input, Tensor output, int kh, int kw,
               int dh, int dw, int ph, int pw, int sh, int sw)
        : PoolingObj(graph, OpType::MaxPool, input, output, kh, kw, dh, dw, ph,
                     pw, sh, sw) {}
};
class AvgPoolObj : public PoolingObj {
  public:
    AvgPoolObj(GraphObj *graph, Tensor input, Tensor output, int kh, int kw,
               int dh, int dw, int ph, int pw, int sh, int sw)
        : PoolingObj(graph, OpType::AvgPool, input, output, kh, kw, dh, dw, ph,
                     pw, sh, sw) {}
};
}; // namespace infini
