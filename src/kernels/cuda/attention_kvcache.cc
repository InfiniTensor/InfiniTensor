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
                    Tensor output_matmul, Tensor output_temp_O, Tensor output_temp_sum) const {
        AttentionKVCacheMetadata metadata;
        initAttentionKVCacheMetadata(metadata, input_v_cache);
        std::cout << "do compute" << std::endl;

        attention_kvcache_kernel(input_k_cache->getRawDataPtr<float *>(),
                                 input_v_cache->getRawDataPtr<float *>(),
                                 input_q->getRawDataPtr<float *>(),
                                 input_k->getRawDataPtr<float *>(),
                                 input_v->getRawDataPtr<float *>(),
                                 position_id->getRawDataPtr<int *>(),
                                 output_matmul->getRawDataPtr<float *>(),
                                 metadata,
                                 output_temp_O->getRawDataPtr<float *>(),
                                 output_temp_sum->getRawDataPtr<float *>());
    }
};

class AttentionKVCacheCuda : private AttentionKVCacheCompute,
                             public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        do_compute(_op->getInputs()[0], _op->getInputs()[1],
                   _op->getInputs()[2], _op->getInputs()[3],
                   _op->getInputs()[4], _op->getInputs()[5],
                   _op->getOutputs()[0], _op->getOutputs()[1], 
                   _op->getOutputs()[2]);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::AttentionKVCache, AttentionKVCacheCuda,
                "AttentionKVCache_CUDA");
} // namespace infini
