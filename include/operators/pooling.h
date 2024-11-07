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
     */
    PoolingObj(GraphObj *graph, OpType optype, Tensor input, Tensor output,
               int kh, int kw, int dh, int dw, int ph, int pw, int sh, int sw,
               int ceilMode);
    OP_CLONE(PoolingObj);

    ~PoolingObj() override {
        if (opDesc) {
            try {
                if (type == OpType::MaxPool) {
                    CHECK_ERROR(infiniopDestroyMaxPoolDescriptor(
                        (infiniopMaxPoolDescriptor_t)opDesc));
                } else if (type == OpType::AveragePool) {
                    CHECK_ERROR(infiniopDestroyAvgPoolDescriptor(
                        (infiniopAvgPoolDescriptor_t)opDesc));
                } else {
                    IT_ASSERT(false, "Unsupported pooling operator type "
                                     "for infini op destroy");
                }
            } catch (const std::exception &e) {
                std::cerr << "Error in ~PoolingObj: " << e.what() << std::endl;
            }
        }
    }

    void initInfiniOp(const Runtime context) override;
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
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

    auto getPadStrideDilation() const { return tuple(ph, pw, sh, sw, dh, dw); }
    auto getNCHWRS() const { return tuple(n, c, h, w, kh, kw); }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

class MaxPoolObj : public PoolingObj {
  public:
    MaxPoolObj(GraphObj *graph, Tensor input, Tensor output, int kh, int kw,
               int dh, int dw, int ph, int pw, int sh, int sw, int ceilMode)
        : PoolingObj(graph, OpType::MaxPool, input, output, kh, kw, dh, dw, ph,
                     pw, sh, sw, ceilMode) {}
};
class AvgPoolObj : public PoolingObj {
  public:
    AvgPoolObj(GraphObj *graph, Tensor input, Tensor output, int kh, int kw,
               int dh, int dw, int ph, int pw, int sh, int sw, int ceilMode)
        : PoolingObj(graph, OpType::AveragePool, input, output, kh, kw, dh, dw,
                     ph, pw, sh, sw, ceilMode) {}
};
}; // namespace infini
