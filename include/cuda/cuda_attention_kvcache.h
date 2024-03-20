#pragma once
#include "core/common.h"
#include <cstdio>

struct AttentionKVCacheMetadata {
    int dimSize[4];
    int stride[4];
};

namespace infini {
void attention_kvcache_kernel(int dType, void *input_k_cache,
                              void *input_v_cache, void *input_q, void *input_k,
                              void *input_v, int *position_id,
                              void *output_matmul,
                              const AttentionKVCacheMetadata &compMeta,
                              float *output_O_temp, float *output_sum_temp);

} // namespace infini
