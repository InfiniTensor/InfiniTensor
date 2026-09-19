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
    int ceilMode;
    int n, c, h, w;
    /// Whether the window is the whole of what the operator is given rather
    /// than a size of its own. A global pool is defined that way, and its
    /// window is therefore not knowable until a real shape arrives.
    bool globalWindow;

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
     * @param ceilMode Whether to use ceil(1) or floor(0) to compute the output
     * shape.
     * @param globalWindow Whether the window is the whole of the input rather
     * than `kh` by `kw`. A global pool passes this, and then `kh` and `kw` are
     * read off each input in turn instead of being fixed here: the spatial size
     * may be dynamic, in which case it is not yet known at construction and
     * whatever is passed would be a placeholder.
     */
    PoolingObj(GraphObj *graph, OpType optype, Tensor input, Tensor output,
               int kh, int kw, int dh, int dw, int ph, int pw, int sh, int sw,
               int ceilMode, bool globalWindow = false);
    OP_CLONE(PoolingObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    /// The window is an attribute rather than an input, so every dimension --
    /// spatial ones included -- follows the one it strides over.
    vector<DimSource> dimSources(size_t output, size_t dim) const override;
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
    int getCeilMode() const { return ceilMode; }
    bool isGlobalWindow() const { return globalWindow; }

    auto getPadStrideDilation() const { return tuple(ph, pw, sh, sw, dh, dw); }
    auto getNCHWRS() const { return tuple(n, c, h, w, kh, kw); }

  private:
    /// Read the batch, channel and spatial sizes off `input`, and the window
    /// too when the window is the whole input. Construction and shape inference
    /// share this so the two cannot come to disagree.
    void takeInputDims(const Tensor &input);

    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

class MaxPoolObj : public PoolingObj {
  public:
    MaxPoolObj(GraphObj *graph, Tensor input, Tensor output, int kh, int kw,
               int dh, int dw, int ph, int pw, int sh, int sw, int ceilMode,
               bool globalWindow = false)
        : PoolingObj(graph, OpType::MaxPool, input, output, kh, kw, dh, dw, ph,
                     pw, sh, sw, ceilMode, globalWindow) {}
};
class AvgPoolObj : public PoolingObj {
  public:
    AvgPoolObj(GraphObj *graph, Tensor input, Tensor output, int kh, int kw,
               int dh, int dw, int ph, int pw, int sh, int sw, int ceilMode,
               bool globalWindow = false)
        : PoolingObj(graph, OpType::AveragePool, input, output, kh, kw, dh, dw,
                     ph, pw, sh, sw, ceilMode, globalWindow) {}
};
}; // namespace infini
