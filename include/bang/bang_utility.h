#include "core/tensor.h"

namespace infini {

void bangPrintFloat(float *x, int len);

void bangPrintTensor(const Tensor &tensor) {
    bangPrintFloat(tensor->getRawDataPtr<float *>(), tensor->size());
}

} // namespace infini
