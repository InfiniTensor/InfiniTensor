#pragma once

#include "operators/unary.h"

namespace infini {
void elu_kernel(const float *input, float *output, int size, float alpha);

} // namespace infini
