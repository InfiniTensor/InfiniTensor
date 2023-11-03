#pragma once
#include "operators/unary.h"
#include "utils/small_array.h"

namespace infini {
void whereKernel(const float *inputX, const float *inputY,
                 const uint8_t *condition, float *output, int nDims,
                 SmallArray inputXShape, SmallArray inputYShape,
                 SmallArray conditionShape, SmallArray outputShape);

}; // namespace infini
