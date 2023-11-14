#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief Fused Attention with KVCache input operator. All the input and output
 * tensors should have the same rank except for the position_id.
 *
 */
class AttentionKVCacheObj : public OperatorObj {
    int dim;

  public:
    /**
     * @brief Construct a new AttentionKVCache object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input_k_cache The k_cache input tensor.
     * @param input_v_cache The v_cache input tensor.
     * @param input_q The query input tensor.
     * @param input_k The key input tensor.
     * @param input_v The value input tensor.
     * @param position_id The positon id of the query,
     * @param output_matmul The query output tensor.
     */
    AttentionKVCacheObj(GraphObj *graph, Tensor input_k_cache,
                        Tensor input_v_cache, Tensor input_q, Tensor input_k,
                        Tensor input_v, Tensor position_id,
                        Tensor output_matmul);
    OP_CLONE(AttentionKVCacheObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    int numInputs() const override { return 6; }
    int numOutputs() const override { return 1; }
    int getDim() const { return dim; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
} // namespace infini
