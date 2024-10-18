#pragma once

#include "operators/unary.h"

namespace infini {

void cumSum_kernel(int dType, void *input, void *weight, void *output,
                    int num_tokens, int hidden_size);

}; // namespace infini
