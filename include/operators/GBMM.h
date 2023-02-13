#pragma once
#include "core/operator.h"
#include <assert.h>
namespace infini {
/**
 * @brief General band matrix multiplication. See
 * https://cscproxy.mpi-magdeburg.mpg.de/mpcsc/benner/pub/brdeq-cle2014.pdf for
 * detail.
 *
 */
class GBMMObj : public OperatorObj {
  private:
    int dilation;
    ActType act;

    int b, m, w, n;

  public:
    /**
     * @brief Construct a new GBMM object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param A The input tensor.
     * @param B The input tensor.
     * @param C C is the output of G2BMM. If outputs are going to be created in
     * the constructor, C should be an empty Ref.
     * @param dilation The dilation of the attention window.
     * @param bias The bias tensor.
     * @param act The activation.
     */
    GBMMObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, const int dilation,
            Tensor bias = nullptr, ActType act = ActType::None);
    OP_CLONE(GBMMObj);

    std::string toString() const override;
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    int numInputs() const override { return 2; }
    int numOutputs() const override { return 1; }

    int getDilation() const { return dilation; }
    Tensor getBias() const { return inputs[2]; }
    ActType getAct() const { return act; }

    int getB() const { return b; }
    int getM() const { return m; }
    int getW() const { return w; }
    int getN() const { return n; }
    auto getBMWND() const { return tuple{b, m, w, n, dilation}; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

} // namespace infini
