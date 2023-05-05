#pragma once

#include "operators/transpose.h"
#include "utils/small_array.h"

namespace infini {

void transpose_kernel(float *input, float *output, int nDims, int size,
                      SmallArray strides, SmallArray outputShape,
                      vector<int> _dims_in, vector<int> _dims_out,
                      vector<int> _perms);

void invoke_transpose_last_two_dim(float *ptrA, float *ptrB, int dim0, int dim1,
                                   int dim2, int numSMs);

} // namespace infini
