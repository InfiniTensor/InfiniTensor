#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "utils/small_array.h"

namespace infini {
void print_test_add();
void whereKernel_py(const float *inputX, const float *inputY,
                 const uint8_t *condition, float *output, int nDims,
                 int outputsize, SmallArray inputXShape, SmallArray inputYShape,
                 SmallArray conditionShape, SmallArray outputShape, int xSize,
                 int ySize, int cSize);

} // namespace infini