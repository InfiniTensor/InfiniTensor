#include <core/tensor_base.h>
namespace infini {

TensorBaseNode::TensorBaseNode(int dim, DataType dtype)
    : dim(dim), dtype(dtype) {}

VType TensorBaseNode::getData(size_t offset) const { return data[offset]; }

}; // namespace infini