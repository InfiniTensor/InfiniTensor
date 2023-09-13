#pragma once
#include "operators/unary.h"
#include "utils/small_array.h"

namespace infini {
void where_kernel(const float *inputX, const float *inputY,
                  const uint8_t *condition, float *output, int nDims,
                  infini::SmallArray inputXShape,
                  infini::SmallArray inputYShape,
                  infini::SmallArray conditionShape,
                  infini::SmallArray outputShape);

}; // namespace infini