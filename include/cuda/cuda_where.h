#pragma once
#include "operators/unary.h"
#include "utils/small_array.h"

namespace infini {
template <typename Tdata>
void whereKernel(const Tdata *inputX, const Tdata *inputY,
                 const uint8_t *condition, Tdata *output, int nDims,
                 int outputsize, SmallArray inputXShape, SmallArray inputYShape,
                 SmallArray conditionShape, SmallArray outputShape, int xSize,
                 int ySize, int cSize);

}; // namespace infini
