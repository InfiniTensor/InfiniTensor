#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief Fused Attention with KVCache input operator.
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
     *          Shape: [batchsize, num_heads, k_cache_seq_length, head_dim]
     * @param input_v_cache The v_cache input tensor.
     *          Shape: [batchsize, num_heads, v_cache_seq_length, head_dim]
     * @param input_q The query input tensor.
     *          Shape: [batchsize, q_seq_length, model_dim]
     * @param input_k The key input tensor.
     *          Shape: [batchsize, q_seq_length, model_dim]
     * @param input_v The value input tensor.
     *          Shape: [batchsize, q_seq_length, model_dim]
     * @param position_id The positon id of the query.
     *          Shape: [batchsize, q_seq_length]
     * @param output_matmul The query output tensor.
     *          Shape: [batchsize, q_seq_length, model_dim]
     */
    AttentionKVCacheObj(GraphObj *graph, Tensor input_k_cache,
                        Tensor input_v_cache, Tensor input_q, Tensor input_k,
                        Tensor input_v, Tensor position_id,
                        Tensor output_matmul);
    OP_CLONE(AttentionKVCacheObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    vector<DataType> inferDataType(const TensorVec &inputs) const override {
        return {inputs[2]->getDType()};
    };
    DataType getDType() const { return getInputs(2)->getDType(); }

    std::string toString() const override;
    int numInputs() const override { return 6; }
    int numOutputs() const override { return 1; }
    int getDim() const { return dim; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
} // namespace infini
