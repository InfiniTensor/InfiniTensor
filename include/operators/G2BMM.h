#pragma once
#include "core/operator.h"
#include <assert.h>
namespace infini {

class G2BMMObj : public OperatorObj {
  private:
    // to be implemented
    int width, dilation;
    ActType act;

    int b, m, k;

  public:
    /**
     * @brief This comments show how operators is defined in InfiniTensor. The
     * constructor can create output tensors for the operator or not, which
     * depends on `graph`.
     *
     * @param graph If graph is not empty, create outputs in the constructor.
     * Otherwise, check the provided shape with the results of `inferShape` in
     * `checkValid`.
     * @param C C is the output of G2BMM. If outputs are going to be created in
     * the constructor, C should be an empty Ref.
     */
    G2BMMObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, const int width,
             const int dilation, Tensor bias = nullptr,
             ActType act = ActType::None);

    std::string toString() const override;
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    int numInputs() const override { return 2; }
    int numOutputs() const override { return 1; }

    int getWidth() const { return width; }
    int getDilation() const { return dilation; }
    Tensor getBias() const { return inputs[2]; }
    ActType getAct() const { return act; }

    int getB() const { return b; }
    int getM() const { return m; }
    int getK() const { return k; }
    auto getBMKWD() const { return tuple{b, m, k, width, dilation}; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

} // namespace infini
