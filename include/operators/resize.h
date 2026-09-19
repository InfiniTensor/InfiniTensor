#pragma once

#include "core/operator.h"

namespace infini {
/**
 * @brief Resize the input tensor. See
 * https://github.com/onnx/onnx/blob/main/docs/Operators.md#Resize for detail.
 *
 */
class ResizeObj : public OperatorObj {
  public:
    enum class ECoordinateTransMode {
        halfPixel,
        pytorchHalfPixel,
        alignCorners,
        asymmetric,
        tfCropAndResize
    };
    enum class ENearestMode {
        roundPreferFloor,
        roundPreferCeil,
        floor,
        ceil,
        none
    };
    enum class EKeepAspectRatioPolicy { stretch, notLarger, notSmaller, none };
    enum class ECoeffMode { nearest, linear, cubic };

    static ECoordinateTransMode fromECoordinateTransModeStr(string mode) {
        if (mode == "half_pixel") {
            return ECoordinateTransMode::halfPixel;
        } else if (mode == "asymmetric") {
            return ECoordinateTransMode::asymmetric;
        } else if (mode == "align_corners") {
            return ECoordinateTransMode::alignCorners;
        } else if (mode == "pytorch_half_pixel") {
            return ECoordinateTransMode::pytorchHalfPixel;
        } else if (mode == "tf_crop_and_resize") {
            return ECoordinateTransMode::tfCropAndResize;
        } else {
            IT_TODO_HALT();
        }
    }

    static ENearestMode fromENearestModeStr(string mode) {
        if (mode == "round_prefer_floor") {
            return ENearestMode::roundPreferFloor;
        } else if (mode == "round_prefer_ceil") {
            return ENearestMode::roundPreferCeil;
        } else if (mode == "floor") {
            return ENearestMode::floor;
        } else if (mode == "ceil") {
            return ENearestMode::ceil;
        } else {
            return ENearestMode::none;
        }
    }

    static EKeepAspectRatioPolicy fromRatioPolicyStr(string ratioPolicyStr) {
        if (ratioPolicyStr == "stretch") {
            return EKeepAspectRatioPolicy::stretch;
        } else if (ratioPolicyStr == "not_larger") {
            return EKeepAspectRatioPolicy::notLarger;
        } else if (ratioPolicyStr == "not_smaller") {
            return EKeepAspectRatioPolicy::notSmaller;
        } else {
            return EKeepAspectRatioPolicy::none;
        }
    }

    static ECoeffMode fromECoeffModeStr(string mode) {
        if (mode == "nearest") {
            return ECoeffMode::nearest;
        } else if (mode == "linear") {
            return ECoeffMode::linear;
        } else if (mode == "cubic") {
            return ECoeffMode::cubic;
        } else {
            IT_TODO_HALT();
        }
    }

  private:
    vector<int> axes;
    vector<float> scales;
    // Output sizes as the model asked for them, one per entry of `axes`, kept
    // only when resizing by sizes. A scale is the ratio of one of these to the
    // input dimension it applies to, so it cannot be worked out once and kept:
    // a dynamic input dimension is a placeholder until a real shape arrives,
    // and a ratio taken against the placeholder resizes every later shape by
    // it. The request is what the model settled; the ratio follows the input.
    vector<int64_t> sizesRequested;
    vector<float> roi;
    bool isGivenSizes = false;

    ECoordinateTransMode coMode; // compute src coordinate from dst coordinate
    ECoeffMode mode; // coeff mode,for computing dst value from coordinate src
                     // neighborhood .
    ENearestMode nearestMode; // used in "nearest" mode, indicates how to get
                              // "nearest" pixel
    EKeepAspectRatioPolicy
        ratioPolicy; // used for computing shape when using "sizes"

  public:
    // nearest mode
    ResizeObj(
        GraphObj *graph, Tensor input, Tensor output,
        const std::optional<vector<int>> &axes, Tensor sizes, Tensor scales,
        Tensor roi,
        EKeepAspectRatioPolicy ratioPolicy = EKeepAspectRatioPolicy::none,
        ENearestMode nearestMode = ENearestMode::roundPreferFloor,
        ECoordinateTransMode coordTransMode = ECoordinateTransMode::halfPixel);

    ResizeObj(
        GraphObj *graph, Tensor input, Tensor output,
        const std::optional<vector<int>> &axes, Tensor sizes, Tensor scales,
        Tensor roi, ECoeffMode mode,
        EKeepAspectRatioPolicy ratioPolicy = EKeepAspectRatioPolicy::none,
        ECoordinateTransMode coordTransMode = ECoordinateTransMode::halfPixel);
    OP_CLONE(ResizeObj);

    // Operator clone(TensorVec inputs, TensorVec outputs) override;
    vector<DataType> inferDataType(const TensorVec &inputs) const override;
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    /// Every dimension keeps its place. One being resized by an explicit size
    /// is that size whatever it was given, so it follows nothing; one resized
    /// by a scale follows the dimension it scales.
    vector<DimSource> dimSources(size_t output, size_t dim) const override;
    std::string toString() const override;
    int numInputs() const override { return inputs.size(); }
    int numOutputs() const override { return 1; }

    ECoeffMode getMode() const { return mode; }
    int getNearestMode() const { return enum_to_underlying(nearestMode); }
    int getKeepAxesRatioPolicy() const {
        return enum_to_underlying(ratioPolicy);
    }
    int getCoordinateTransMode() const { return enum_to_underlying(coMode); }
    float getScale(int i) const {
        IT_ASSERT((size_t)i < scales.size());
        return scales.at(i);
    }

    vector<float> getScales() const { return scales; }

    float getRoi(int i) const {
        if (coMode == ECoordinateTransMode::tfCropAndResize) {
            IT_ASSERT(size_t(i) < roi.size());
            return roi.at(i);
        } else
            return 0;
    }
    void setGivenSizes(bool val) { isGivenSizes = val; }
    bool isResizeBySizes() const { return isGivenSizes; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;

    float round_int(float x) const;
    void init(const Tensor &input, const Tensor &sizes, const Tensor &scales,
              const Tensor &roi, const std::optional<vector<int>> &axes);
    void InitBySizes(Tensor input, Tensor sizes,
                     const std::optional<vector<int>> &axes);
    /// Works the scales out afresh from `sizesRequested` against the shape
    /// `input` carries now. Called whenever shapes are inferred, so that the
    /// scales the kernels read describe the current input rather than the one
    /// construction happened to see.
    void takeScalesFromSizes(const Tensor &input);
    void InitByScales(Tensor input, Tensor sizes,
                      const std::optional<vector<int>> &axes);
};
} // namespace infini
