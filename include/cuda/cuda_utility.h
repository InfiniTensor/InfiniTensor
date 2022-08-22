#include "core/tensor.h"

namespace infini {

void cudaPrintFloat(float *x, int len);

void cudaPrintTensor(const Tensor &tensor) {
    cudaPrintFloat(tensor->getDataRawPtr<float *>(), tensor->size());
}

} // namespace infini