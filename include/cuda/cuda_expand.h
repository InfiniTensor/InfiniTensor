#pragma once

#include "operators/unary.h"

namespace infini {
void expand_kernel(float *d_input, float *d_output, int shape, int inputsize);

}; // namespace infini
