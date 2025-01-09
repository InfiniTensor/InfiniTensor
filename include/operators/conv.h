#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief Convolution. Currently this operator only supports 2-D convolution.
 * This is the base class for convolution and transposed convolution.
 * The input tensor has four dimensions, called N (batch), C (channel), H
 * (height), and W (width) respectively; The weight tensor has four dimensions,
 * called F (number of filters), C (channel), R (height of weight), and S (width
 * of weight) respectively; The output tensor has four dimensions, called N, F,
 * H, and W respectively. By default, we take NCHW layout for the input and
 * output tensors, and FCRS layout for the weight tensor.
 * Convolutions have three attributes, called padding, stride, and dilation.
 * Padding is assigned by padding mode or padding size. Padding mode must be
 * Other, Same, or Valid (see the definition of enum class PaddingMode). Same
 * means the output has the same shape as the input. Valid means padding size is
 * 0. Other means padding size is assigned by value ph and pw, denoting the
 * padding size along height dimension and weight dimension, respectively.
 * Stride is assigned by sh and sw, denoting the stride along height dimension
 * and weight dimension, respectively.
 * Dilation is assigned by dh and dw, denoting the dilation along height
 * dimension and weight dimension, respectively.
 * See
 * https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d
 * for a detailed explanation of convolution.
 *
 */
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
    // Auxiliary attributes. Descripitions stand on a forward perspective,
    // i.e., convTransposed2d is not regarded as the backward of conv2d.
    int n;    // batch size
    int c;    // input/output channel for conv2d/convTransposed2d
    int h, w; // input shape (same for conv2d and convTranposed2d)
    int f;    // output/input channel for conv2d/convTransposed2d
    int r, s; // weight shape

    ActType act;

  public:
    /**
     * @brief Construct a new ConvBase object by explicitly setting padding
     * size.
     *
     * @param opType Indicate if this is a convolution or transposed
     * convolution.
     * @param inputs The input, weight and bias tensors. Bias is optional.
     * FIXME: Split inputs into three parameters, input, weight, and bias.
     * @param output The output tensor.
     * @param ph Padding along height dimension.
     * @param pw Padding along weight dimension.
     * @param sh Stride along height dimension.
     * @param sw Stride along weight dimension.
     * @param dh Dilation along height dimension.
     * @param dw Dilation along weight dimension.
     * @param inputInConvFWD To be removed.
     * @param weightInConvFWD To be removed.
     */
    ConvBaseObj(OpType opType, TensorVec inputs, Tensor &output, int ph, int pw,
                int sh, int sw, int dh, int dw, const Tensor &inputInConvFWD,
                const Tensor &weightInConvFWD, ActType act = ActType::None);
    /**
     * @brief Construct a new ConvBase object by setting padding mode.
     *
     * @param opType Indicate if this is a convolution or transposed
     * convolution.
     * @param inputs The input, weight and bias tensors. Bias is optional.
     * FIXME: Split inputs into three parameters, input, weight, and bias.
     * @param output The output tensor.
     * @param mode Padding mode, which is set to Other, Same, or Valid.
     * @param sh Stride along height dimension.
     * @param sw Stride along weight dimension.
     * @param dh Dilation along height dimension.
     * @param dw Dilation along weight dimension.
     * @param inputInConvFWD To be removed.
     * @param weightInConvFWD To be removed.
     */
    ConvBaseObj(OpType opType, TensorVec inputs, Tensor &output,
                PaddingMode mode, int sh, int sw, int dh, int dw,
                const Tensor &inputInConvFWD, const Tensor &weightInConvFWD,
                ActType act = ActType::None);

    std::string toString() const override;
    int numInputs() const override { return inputs.size(); }
    int numOutputs() const override { return outputs.size(); }

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
    int getChannelPerGroup() const {
        if (type == OpType::ConvTransNHWC) {
            return inputs[1]->getDims()[3];
        } else {
            return inputs[1]->getDims()[1];
        }
    }
    ActType getAct() const { return act; }
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
  public:
    ConvObj(GraphObj *graph, Tensor input, Tensor weight, Tensor output, int ph,
            int pw, Tensor bias = nullptr, int sh = 1, int sw = 1, int dh = 1,
            int dw = 1, ActType act = ActType::None);
    // Constructors for setting padding mode
    ConvObj(GraphObj *graph, Tensor input, Tensor weight, Tensor output,
            Tensor bias = nullptr, PaddingMode mode = PaddingMode::Same,
            int sh = 1, int sw = 1, int dh = 1, int dw = 1,
            ActType act = ActType::None);
    OP_CLONE(ConvObj);

    ~ConvObj() override {
        if (opDesc) {
            try {
                // if (numInputs() == 2) {
                //     CHECK_ERROR(infiniopDestroyConvDescriptor(
                //         (infiniopConvDescriptor_t)opDesc));
                // } else if (numInputs() == 3) {
                //     CHECK_ERROR(infiniopDestroyConvBiasActDescriptor(
                //         (infiniopConvBiasActDescriptor_t)opDesc));
                // }
            } catch (const std::exception &e) {
                std::cerr << "Error in ~ConvObj: " << e.what() << std::endl;
            }
        }
    }

    void initInfiniOp(const Runtime context) override;
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    int getNumGroups() const override { return c / getChannelPerGroup(); }

  private:
    void setAuxilaryAttributes(PaddingMode mode) override;
};

class Conv3dObj : public ConvBaseObj {
  protected:
    int pd;
    int sd;
    int dd;
    // Auxiliary attributes.
    int d; // Input depth.
    int q; // Weight depth.

  public:
    Conv3dObj(GraphObj *graph, Tensor input, Tensor weight, Tensor output,
              int pd, int ph, int pw, int sd = 1, int sh = 1, int sw = 1,
              int dd = 1, int dh = 1, int dw = 1, Tensor bias = nullptr,
              ActType act = ActType::None);
    // Constructors for setting padding mode.
    Conv3dObj(GraphObj *graph, Tensor input, Tensor weight, Tensor output,
              PaddingMode mode = PaddingMode::Same, int sd = 1, int sh = 1,
              int sw = 1, int dd = 1, int dh = 1, int dw = 1,
              Tensor bias = nullptr, ActType act = ActType::None);
    OP_CLONE(Conv3dObj);

    std::string toString() const override;
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    int getNumGroups() const override { return c / getChannelPerGroup(); }

    auto getNCDHWFQRS() const { return tuple(n, c, d, h, w, f, q, r, s); }
    auto getPadStrideDilation() const {
        return tuple(pd, ph, pw, sd, sh, sw, dd, dh, dw);
    }

  private:
    void setAuxilaryAttributes(PaddingMode mode) override;
};

class ConvBackwardFilterObj : public ConvBaseObj {
  private:
    ActType act;

  public:
    ConvBackwardFilterObj(GraphObj *graph, Tensor inputX, Tensor diffY,
                          Tensor diffW, int ph, int pw, int sh = 1, int sw = 1,
                          int dh = 1, int dw = 1, Tensor bias = nullptr,
                          ActType act = ActType::None);
    // Constructors for setting padding mode
    ConvBackwardFilterObj(GraphObj *graph, Tensor inputX, Tensor diffY,
                          Tensor diffW, PaddingMode mode = PaddingMode::Same,
                          int sh = 1, int sw = 1, int dh = 1, int dw = 1,
                          Tensor bias = nullptr, ActType act = ActType::None);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    ActType getAct() const { return act; }
    int getNumGroups() const override { return c / getChannelPerGroup(); }

  private:
    void setAuxilaryAttributes(PaddingMode mode) override;
};

class ConvTransposed2dObj : public ConvBaseObj {
  private:
    int oph, opw;
    int group;

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
    OP_CLONE(ConvTransposed2dObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    int getNumGroups() const override { return group; }
    std::pair<int, int> getOutputPadding() const { return {oph, opw}; }

  private:
    void setAuxilaryAttributes(PaddingMode mode) override;
};

class ConvTransposed2dNHWCObj : public ConvBaseObj {
  private:
    int oph, opw;
    int group;

  public:
    ConvTransposed2dNHWCObj(GraphObj *graph, Tensor input, Tensor weight,
                            Tensor output, int ph, int pw, int sh = 1,
                            int sw = 1, int dh = 1, int dw = 1, int oph = 0,
                            int opw = 0, int group = 1, Tensor bias = nullptr,
                            ActType act = ActType::None);
    // Constructors for setting padding mode
    ConvTransposed2dNHWCObj(GraphObj *graph, Tensor input, Tensor weight,
                            Tensor output, PaddingMode mode = PaddingMode::Same,
                            int sh = 1, int sw = 1, int dh = 1, int dw = 1,
                            int oph = 0, int opw = 0, int group = 1,
                            Tensor bias = nullptr, ActType act = ActType::None);
    OP_CLONE(ConvTransposed2dNHWCObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    int getNumGroups() const override { return group; }

  private:
    void setAuxilaryAttributes(PaddingMode mode) override;
};

} // namespace infini
