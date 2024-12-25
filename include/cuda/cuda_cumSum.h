#pragma once

#include "operators/unary.h"

namespace infini {

void cumSum_kernel(const Tensor& input, int axis, Tensor& output, 
                    bool exclusive, bool reverse);

}; // namespace infini