#pragma once
#include "operators/unary.h"
#include "utils/small_array.h"
namespace infini {

void whereKernel(int dTypeIndex, void *inputX, void *inputY,
                 const uint8_t *condition, void *output, int a0, int a1, int a2,
                 int a3, int b0, int b1, int b2, int b3, int c0, int c1, int c2,
                 int c3, int d0, int d1, int d2, int d3);
void whereKernel(int dTypeIndex, void *inputX, void *inputY,
                 const uint8_t *condition, void *output, int nDims,
                 int outputsize, SmallArray inputXShape, SmallArray inputYShape,
                 SmallArray conditionShape, SmallArray outputShape, int xSize,
                 int ySize, int cSize);
}; // namespace infini
