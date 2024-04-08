#pragma once
#include "core/common.h"
#include <cstdio>

struct AttentionKVCacheMetadata {
    int head_dim;
    int num_heads;
    int num_seqs;
    int max_kv_seqlen;
};

namespace infini {
void attention_kvcache_kernel(int dType, void *input_k_cache,
                              void *input_v_cache, void *input_q, void *input_k,
                              void *input_v, int64_t *position_id,
                              void *output_matmul,
                              const AttentionKVCacheMetadata &compMeta,
                              float *output_O_temp, float *output_sum_temp);

} // namespace infini
