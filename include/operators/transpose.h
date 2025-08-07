#pragma once
#include "core/operator.h"

namespace infini {
class TransposeObj : public OperatorObj {
  public:
    TransposeObj(GraphObj *graph, Tensor input, Tensor output, Shape permute);
    OP_CLONE(TransposeObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }
    Shape getPermute() const { return transposePermute; }

  private:
    Shape transposePermute;
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

class DepthToSpaceObj : public OperatorObj {
  public:
    DepthToSpaceObj(GraphObj *graph, Tensor input, Tensor output, int blocksize,
                    std::string mode);
    OP_CLONE(DepthToSpaceObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }
    int getBlockSize() const { return blockSize; }
    int getMode() const { return D2SMode; }
    auto getModeString() const { return D2SModeString; }
    auto getReshapeDim() const { return reshapeDim; }
    auto getTransposeDim() const { return transposeDim; }
    auto getOutDim() const { return outDim; }

  private:
    int blockSize;
    int D2SMode;
    std::string D2SModeString;
    mutable std::vector<size_t> reshapeDim = {1, 1, 1, 1, 1, 1};
    mutable std::vector<size_t> transposeDim = {1, 1, 1, 1, 1, 1};
    mutable std::vector<size_t> outDim = {1, 1, 1, 1};
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

} // namespace infini
