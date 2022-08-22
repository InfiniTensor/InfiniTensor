#include "core/tensor_base.h"
#include "core/blob.h"
#include "core/runtime.h"
namespace infini {

TensorBaseObj::TensorBaseObj(int dim, DataType dtype)
    : dim(dim), dtype(dtype) {}

VType TensorBaseObj::getData(size_t offset) const {
    // TODO: check cuda array
    return (data->getPtr<VType *>())[offset];
}

}; // namespace infini