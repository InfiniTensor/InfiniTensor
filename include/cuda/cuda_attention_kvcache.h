#pragma once
#include "core/common.h"
#include <cstdio>

struct AttentionKVCacheMetadata {
    int dimSize[4];
    int stride[4];
};

namespace infini {
void attention_kvcache_kernel(float *input_k_cache, float *input_v_cache,
                              float *input_q, float *input_k, float *input_v,
                              int *position_id, float *output_matmul,
                              const AttentionKVCacheMetadata &compMeta,
                              float *output_O_temp, float *output_sum_temp);

} // namespace infini
