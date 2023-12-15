#pragma once
#include "operators/unary.h"
#include "utils/small_array.h"

namespace infini {

void whereKernel(int dTypeIndex, void *inputX, void *inputY,
                 const uint8_t *condition, void *output, int nDims,
                 int outputsize, SmallArray inputXShape, SmallArray inputYShape,
                 SmallArray conditionShape, SmallArray outputShape, int xSize,
                 int ySize, int cSize);
}; // namespace infini
