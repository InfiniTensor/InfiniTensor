#pragma once
#include "core/tensor.h"
#include "cuda/cuda_common.h"

namespace infini {

void cudaPrintFloat(float *x, int len);

void cudaPrintTensor(const Tensor &tensor);

cudnnDataType_t cudnnDataTypeConvert(DataType dataType);

} // namespace infini
