#pragma once

#include "core/operator.h"

namespace infini {
class ResizeObj : public OperatorObj {
  public:
    enum class ECoordinateTransMode {
        halfPixel,
        pytorchHalfPixel,
        alignCorners,
        asymmetric,
        tfCropAndResize
    };
    enum class ENearestMode { roundPreferFloor, roundPreferCeil, floor, ceil };
    enum class EKeepAspectRatioPolicy { stretch, notLarger, notSmaller };
    enum class ECoeffMode { nearest, linear, cubic };

  private:
    vector<int> axes;
    vector<float> scales;
    ECoordinateTransMode coMode; // compute src coordinate from dst coordinate
    ECoeffMode mode; // coeff mode,for computing dst value from coordinate src
                     // neighborhood .
    ENearestMode nearestMode; // used in "nearest" mode, indicates how to get
                              // "nearest" pixel
    EKeepAspectRatioPolicy
        ratioPolicy; // used for computing shape when using "sizes"

  public:
    // nearest mode, not tf_crop_and_resize
    ResizeObj(
        GraphObj *graph, Tensor input, Tensor output,
        const std::optional<vector<int>> &axes, Tensor sizes,
        EKeepAspectRatioPolicy ratioPolicy,
        ENearestMode nearestMode = ENearestMode::roundPreferFloor,
        ECoordinateTransMode coordTransMode = ECoordinateTransMode::halfPixel);
    ResizeObj(
        GraphObj *graph, Tensor input, Tensor output,
        const std::optional<vector<int>> &axes, Tensor scales,
        ENearestMode nearestMode = ENearestMode::roundPreferFloor,
        ECoordinateTransMode coordTransMode = ECoordinateTransMode::halfPixel);

    // linear mode
    ResizeObj(
        GraphObj *graph, Tensor input, Tensor output,
        const std::optional<vector<int>> &axes, Tensor sizes,
        EKeepAspectRatioPolicy ratioPolicy, ECoeffMode mode,
        ECoordinateTransMode coordTransMode = ECoordinateTransMode::halfPixel);
    ResizeObj(
        GraphObj *graph, Tensor input, Tensor output,
        const std::optional<vector<int>> &axes, Tensor scales, ECoeffMode mode,
        ECoordinateTransMode coordTransMode = ECoordinateTransMode::halfPixel);

    vector<DataType> inferDataType(const TensorVec &inputs) const override;
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;
    std::string toString() const override;
    int numInputs() const override { return 4; }
    int numOutputs() const override { return 1; }

    ECoeffMode getMode() const { return mode; }
    int getNearestMode() const { return enum_to_underlying(nearestMode); }
    int getKeepAxesRatioPolicy() const {
        return enum_to_underlying(ratioPolicy);
    }
    int getCoordinateTransMode() const { return enum_to_underlying(coMode); }
    float getScale(int i) const {
        IT_ASSERT((size_t)i < scales.size());
        return scales[i];
    }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;

    float round_int(float x) const;
    bool checkCoordinateTransValid(int resizedCo, int origiCo) const;
    void InitBySizes(Tensor input, Tensor sizes,
                     const std::optional<vector<int>> &axes);
    void InitByScales(Tensor input, Tensor sizes,
                      const std::optional<vector<int>> &axes);
};
} // namespace infini
