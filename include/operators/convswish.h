#pragma once
#include "core/operator.h"

namespace infini {
class ConvSwishObj : public OperatorObj {
    int n;    // batch size
    int c;    // input/output channel for conv2d/convTransposed2d
    int h, w; // input shape (same for conv2d and convTranposed2d)
    int f;    // output/input channel for conv2d/convTransposed2d
    int r, s; // weight shape
    vector<int> pad;
    vector<int> stride;
    vector<int> dilation;

  public:
    ConvSwishObj(GraphObj *graph, Tensor input, Tensor weight, Tensor bias,
                 Tensor output, vector<int> pad, vector<int> stride,
                 vector<int> dilation);
    OP_CLONE(ConvSwishObj);

    std::string toString() const override;
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    int numInputs() const override { return inputs.size(); }
    int numOutputs() const override { return 1; }
    auto getNCHWFRS() const { return tuple(n, c, h, w, f, r, s); }
    auto getPadStrideDilation() const {
        return tuple(pad[0], pad[1], stride[0], stride[1], dilation[0],
                     dilation[1]);
    }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
} // namespace infini
