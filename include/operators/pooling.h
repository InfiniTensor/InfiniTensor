#pragma once
#include "core/operator.h"

namespace infini {

class PoolingObj : public OperatorObj {
  private:
    int kh, kw;
    int dh, dw;
    int ph, pw;
    int sh, sw;
    int n, c, h, w;

  public:
    PoolingObj(GraphObj *graph, OpType optype, Tensor input, Tensor output,
               int kh, int kw, int dh, int dw, int ph, int pw, int sh, int sw);

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
    virtual std::string getOpName() const = 0;
};

class MaxPoolObj : public PoolingObj {
  public:
    MaxPoolObj(GraphObj *graph, Tensor input, Tensor output, int kh, int kw,
               int dh, int dw, int ph, int pw, int sh, int sw)
        : PoolingObj(graph, OpType::MaxPool, input, output, kh, kw, dh, dw, ph,
                     pw, sh, sw) {}

  private:
    std::string getOpName() const override { return "Maxpool"; }
};
class AvgPoolObj : public PoolingObj {
  public:
    AvgPoolObj(GraphObj *graph, Tensor input, Tensor output, int kh, int kw,
               int dh, int dw, int ph, int pw, int sh, int sw)
        : PoolingObj(graph, OpType::AvgPool, input, output, kh, kw, dh, dw, ph,
                     pw, sh, sw) {}

  private:
    std::string getOpName() const override { return "Avgpool"; }
};
}; // namespace infini