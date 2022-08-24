#include "core/tensor.h"

namespace infini {

void cudaPrintFloat(float *x, int len);

void cudaPrintTensor(const Tensor &tensor) {
    cudaPrintFloat(tensor->getRawDataPtr<float *>(), tensor->size());
}

} // namespace infini