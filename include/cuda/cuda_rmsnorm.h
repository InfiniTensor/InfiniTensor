#pragma once

#include "operators/rms_norm.h"

namespace infini {

void rmsnorm_kernel(int dType, void *input, void *weight, void *output,
                    int num_tokens, int hidden_size);

}; // namespace infini
