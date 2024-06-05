#pragma once

#include "operators/unary.h"

namespace infini {

void leaky_relu_kernel(float *input, float *output, float alphaValue, int size);

}; // namespace infini