#pragma once
#include "core/operator.h"
#include "core/kernel.h"
#include "cuda/cuda_runtime.h"
#include <chrono>
#include <functional>
#include <limits>
#include <tuple>

namespace infini {

class ConvBaseObj : public OperatorObj {
  public:
    // When PaddingMode is Other, ConvObj will use padding size (ph, pw)
    // Otherwise, padding size (ph, pw) will be computed by padding mode
    enum class PaddingMode {
        Other,
        Same,
        Valid,
    };

  protected:
    int ph, pw;
    int sh, sw;
    int dh, dw;
    PaddingMode padding;
    // auxiliary attributes. Descripitions stand on a forward perspective,
    // i.e., convTransposed2d is not regarded as the backward of conv2d.
    int n;    // batch size
    int c;    // input/output channel for conv2d/convTransposed2d
    int h, w; // input shape (same for conv2d and convTranposed2d)
    int f;    // output/input channel for conv2d/convTransposed2d
    int r, s; // weight shape

  public:
    // Constructors for explicitly setting padding size
    ConvBaseObj(OpType opType, TensorVec inputs, Tensor &output, int ph, int pw,
                int sh, int sw, int dh, int dw, const Tensor &inputInConvFWD,
                const Tensor &weightInConvFWD);
    ConvBaseObj(OpType opType, TensorVec inputs, Tensor &output,
                PaddingMode mode, int sh, int sw, int dh, int dw,
                const Tensor &inputInConvFWD, const Tensor &weightInConvFWD);

    std::string toString() const override;
    int numInputs() const override { return 2; }
    int numOutputs() const override { return 1; }

    Tensor getBias() const { return inputs[2]; }
    PaddingMode getPaddingMode() const { return padding; }
    pair<int, int> inferPaddingSize() const;

    int getDh() const { return dh; }
    int getDw() const { return dw; }
    int getPh() const { return ph; }
    int getPw() const { return pw; }
    int getSh() const { return sh; }
    int getSw() const { return sw; }
    auto getNCHWFRS() const { return tuple(n, c, h, w, f, r, s); }
    auto getPadStrideDilation() const { return tuple(ph, pw, sh, sw, dh, dw); }
    int getChannelPerGroup() const { return inputs[1]->getDims()[1]; }
    virtual int getNumGroups() const = 0;

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
    /**
     * @brief Set the Auxilary Attributes: nchwrfs and padding (ph, pw) if
     * padding mode is set. This function should be called in constructor.
     */
    virtual void setAuxilaryAttributes(PaddingMode mode) = 0;
};

class ConvObj : public ConvBaseObj {
  private:
    ActType act;

  public:
    ConvObj(GraphObj *graph, Tensor input, Tensor weight, Tensor output, int ph,
            int pw, int sh = 1, int sw = 1, int dh = 1, int dw = 1,
            Tensor bias = nullptr, ActType act = ActType::None);
    // Constructors for setting padding mode
    ConvObj(GraphObj *graph, Tensor input, Tensor weight, Tensor output,
            PaddingMode mode = PaddingMode::Same, int sh = 1, int sw = 1,
            int dh = 1, int dw = 1, Tensor bias = nullptr,
            ActType act = ActType::None);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;
    ActType getAct() const { return act; }
    int getNumGroups() const override { return c / getChannelPerGroup(); }

  private:
    void setAuxilaryAttributes(PaddingMode mode) override;
};

class ConvTransposed2dObj : public ConvBaseObj {
  private:
    int oph, opw;
    int group;
    ActType act;

  public:
    ConvTransposed2dObj(GraphObj *graph, Tensor input, Tensor weight,
                        Tensor output, int ph, int pw, int sh = 1, int sw = 1,
                        int dh = 1, int dw = 1, int oph = 0, int opw = 0,
                        int group = 1, Tensor bias = nullptr,
                        ActType act = ActType::None);
    // Constructors for setting padding mode
    ConvTransposed2dObj(GraphObj *graph, Tensor input, Tensor weight,
                        Tensor output, PaddingMode mode = PaddingMode::Same,
                        int sh = 1, int sw = 1, int dh = 1, int dw = 1,
                        int oph = 0, int opw = 0, int group = 1,
                        Tensor bias = nullptr, ActType act = ActType::None);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;
    ActType getAct() const { return act; }
    int getNumGroups() const override { return group; }

  private:
    void setAuxilaryAttributes(PaddingMode mode) override;
};

} // namespace infini
