#pragma once
#include "core/operator.h"

namespace infini {

class ConvObj : public OperatorObj {
  public:
    // When PaddingMode is Other, ConvObj will use padding size (ph, pw)
    // Otherwise, padding size (ph, pw) will be computed by padding mode
    enum class PaddingMode {
        Other,
        Same,
        Valid,
    };

  private:
    int ph, pw;
    int sh, sw;
    int dh, dw;
    ActType act;
    PaddingMode padding;

  public:
    // Constructors for explicitly setting padding size
    ConvObj(GraphObj *graph, Tensor input, Tensor weight, Tensor output, int ph,
            int pw, int sh = 1, int sw = 1, int dh = 1, int dw = 1,
            Tensor bias = nullptr, ActType act = ActType::None);
    // Constructors for setting padding mode
    ConvObj(GraphObj *graph, Tensor input, Tensor weight, Tensor output,
            PaddingMode pm = PaddingMode::Same, int sh = 1, int sw = 1,
            int dh = 1, int dw = 1, Tensor bias = nullptr,
            ActType act = ActType::None);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    int numInputs() const override { return 3; }
    int numOutputs() const override { return 1; }

    Tensor getBias() const { return inputs[2]; }
    ActType getAct() const { return act; }
    PaddingMode getPaddingMode() const { return padding; }
    pair<int, int> inferPaddingSize() const;

    int getDh() const { return dh; }
    int getDw() const { return dw; }
    int getPh() const { return ph; }
    int getPw() const { return pw; }
    int getSh() const { return sh; }
    int getSw() const { return sw; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

} // namespace infini
