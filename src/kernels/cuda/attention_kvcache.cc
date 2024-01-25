#include "operators/attention_kvcache.h"
#include "cuda/cuda_attention_kvcache.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include <functional>

namespace infini {

class AttentionKVCacheCompute {
    void initAttentionKVCacheMetadata(AttentionKVCacheMetadata &metadata,
                                      Tensor tensor) const {
        int nDims = tensor->getRank();
        auto strides = tensor->getStride();
        IT_ASSERT(nDims == 4);
        IT_ASSERT(strides.size() == (size_t)nDims);
        for (int i = 0; i < nDims; ++i) {
            metadata.dimSize[i] = tensor->getDims().at(i);
            metadata.stride[i] = strides.at(i);
        }
    }

  public:
    void do_compute(Tensor input_k_cache, Tensor input_v_cache, Tensor input_q,
                    Tensor input_k, Tensor input_v, Tensor position_id,
                    Tensor output_matmul, CudaPtr p_workspace) const {
        AttentionKVCacheMetadata metadata;
        initAttentionKVCacheMetadata(metadata, input_v_cache);

        attention_kvcache_kernel(input_k_cache->getRawDataPtr<float *>(),
                                 input_v_cache->getRawDataPtr<float *>(),
                                 input_q->getRawDataPtr<float *>(),
                                 input_k->getRawDataPtr<float *>(),
                                 input_v->getRawDataPtr<float *>(),
                                 position_id->getRawDataPtr<int *>(),
                                 output_matmul->getRawDataPtr<float *>(),
                                 metadata, (float *)p_workspace,
                                 (float *)(p_workspace + (1ll << 30)));
    }
};

class AttentionKVCacheCuda : private AttentionKVCacheCompute,
                             public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        IT_ASSERT(_op->getDType() == DataType::Float32);

        size_t workspaceSize = 2ll << 30;
        auto context = dynamic_cast<const CudaRuntimeObj *>(_context);
        CudaPtr idxWsData = context->getWorkspace(workspaceSize);
        do_compute(_op->getInputs()[0], _op->getInputs()[1],
                   _op->getInputs()[2], _op->getInputs()[3],
                   _op->getInputs()[4], _op->getInputs()[5],
                   _op->getOutputs()[0], idxWsData);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::AttentionKVCache, AttentionKVCacheCuda,
                "AttentionKVCache_CUDA");
} // namespace infini
