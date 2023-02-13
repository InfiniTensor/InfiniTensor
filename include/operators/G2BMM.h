#pragma once
#include "core/operator.h"
#include <assert.h>
namespace infini {
/**
 * @brief General to band matrix multiplication, which is used for Longformer
 * model. See https://arxiv.org/pdf/2004.05150.pdf for detail.
 *
 */
class G2BMMObj : public OperatorObj {
  private:
    // to be implemented
    int width, dilation;
    ActType act;

    int b, m, k;

  public:
    /**
     * @brief Construct a new G2BMM object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param A The input tensor.
     * @param B The input tensor.
     * @param C C is the output of G2BMM. If outputs are going to be created in
     * the constructor, C should be an empty Ref.
     * @param width The width of the attention window.
     * @param dilation The dilation of the attention window.
     * @param bias The bias tensor.
     * @param act The activation.
     */
    G2BMMObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, const int width,
             const int dilation, Tensor bias = nullptr,
             ActType act = ActType::None);
    OP_CLONE(G2BMMObj);

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
