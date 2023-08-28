#pragma once
#include "operators/unary.h"
#include "utils/small_array.h"

namespace infini {
void where_kernel(const float *inputx, const float *inputy,
                  const float *condition, float *output, int nDims,
                  infini::SmallArray inputxShape,
                  infini::SmallArray inputyShape,
                  infini::SmallArray conditionShape,
                  infini::SmallArray outputShape);

}; // namespace infini
