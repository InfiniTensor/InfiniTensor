#include "operators/attention_kvcache.h"
#include "cuda/cuda_attention_kvcache.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include <functional>

namespace infini {

class AttentionKVCacheCompute {
    void initAttentionKVCacheMetadata(AttentionKVCacheMetadata &metadata,
                                      Tensor input_v_cache,
                                      Tensor position_id) const {
        int nDims = input_v_cache->getRank();
        auto strides = input_v_cache->getStride();
        IT_ASSERT(nDims == 4);
        int dim_position_id = position_id->getRank();
        metadata.num_seqs = 1;
        for (int i = 0; i < dim_position_id; i++) {
            metadata.num_seqs *= position_id->getDims().at(i);
        }
        metadata.head_dim = input_v_cache->getDims().at(3);
        metadata.num_heads = input_v_cache->getDims().at(1);
        metadata.max_kv_seqlen = input_v_cache->getDims().at(2);
    }

  public:
    void do_compute(int dType, Tensor input_k_cache, Tensor input_v_cache,
                    Tensor input_q, Tensor input_k, Tensor input_v,
                    Tensor position_id, Tensor output_matmul,
                    CudaPtr p_workspace) const {
        AttentionKVCacheMetadata metadata;
        initAttentionKVCacheMetadata(metadata, input_v_cache, position_id);

        attention_kvcache_kernel(
            dType, input_k_cache->getRawDataPtr<void *>(),
            input_v_cache->getRawDataPtr<void *>(),
            input_q->getRawDataPtr<void *>(), input_k->getRawDataPtr<void *>(),
            input_v->getRawDataPtr<void *>(),
            position_id->getRawDataPtr<int64_t *>(),
            output_matmul->getRawDataPtr<void *>(), metadata,
            (float *)p_workspace, (float *)(p_workspace + (1ll << 30)));
    }
};

class AttentionKVCacheCuda : private AttentionKVCacheCompute,
                             public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<AttentionKVCacheObj>(_op);
        int dType = op->getDType().getIndex();
        int position_idx_dtype = op->getInputs()[5]->getDTypeIndex();
        IT_ASSERT(dType == 1 || dType == 10 || position_idx_dtype == 7);

        size_t workspaceSize = 2ll << 30;
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        CudaPtr idxWsData = context->getWorkspace(workspaceSize);
        do_compute(dType, op->getInputs()[0], op->getInputs()[1],
                   op->getInputs()[2], op->getInputs()[3], op->getInputs()[4],
                   op->getInputs()[5], op->getOutputs()[0], idxWsData);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::AttentionKVCache, AttentionKVCacheCuda,
                "AttentionKVCache_CUDA");
} // namespace infini
