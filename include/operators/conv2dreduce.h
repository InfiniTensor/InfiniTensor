#pragma once
#include "core/operator.h"

namespace infini {

class Conv2dReduceBase : public OperatorObj {
  protected:
    Tensor bias;
    int ph, pw;
    int sh, sw;
    int dh, dw;
    int n, h, w, f, r, s; // c has been reduced
    bool PReLU;
    float paramReLU;

  public:
    Conv2dReduceBase(OpType opType, Tensor input, Tensor bias, Tensor output,
                     bool PReLU_, float paramReLU_, int ph_, int pw_,
                     int sh_ = 1, int sw_ = 1, int dh_ = 1, int dw_ = 1);

    std::string toString() const override;
    int numInputs() const override { return 2; }
    int numOutputs() const override { return 1; }

    int getDh() const { return dh; }
    int getDw() const { return dw; }
    int getPh() const { return ph; }
    int getPw() const { return pw; }
    int getSh() const { return sh; }
    int getSw() const { return sw; }
    bool getPReLU() const { return PReLU; }
    float getParamReLU() const { return paramReLU; }

    Tensor getBias() const { return bias; }

    // optional<vector<Shape>> inferShape(const TensorVec &inputs) const
    // override;

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

class Conv2dReduce : public Conv2dReduceBase {
  public:
    Conv2dReduce(GraphObj *graph, Tensor input, Tensor bias, Tensor output,
                 bool PReLU_, float paramReLU_, int ph_, int pw_, int sh_ = 1,
                 int sw_ = 1, int dh_ = 1, int dw_ = 1);
    OP_CLONE(Conv2dReduce);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;
};

class Conv2dReduceTranspose : public Conv2dReduceBase {
  public:
    Conv2dReduceTranspose(GraphObj *graph, Tensor input, Tensor bias,
                          Tensor output, bool PReLU_, float paramReLU_, int ph_,
                          int pw_, int sh_ = 1, int sw_ = 1, int dh_ = 1,
                          int dw_ = 1);
    OP_CLONE(Conv2dReduceTranspose);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;
};
} // namespace infini