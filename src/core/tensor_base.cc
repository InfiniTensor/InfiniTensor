#include <core/tensor_base.h>
namespace infini {

TensorBaseObj::TensorBaseObj(int dim, DataType dtype)
    : dim(dim), dtype(dtype) {}

VType TensorBaseObj::getData(size_t offset) const { return data[offset]; }

}; // namespace infini