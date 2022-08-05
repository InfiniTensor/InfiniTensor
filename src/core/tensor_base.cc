#include <core/tensor_base.h>
namespace it {

TensorBaseNode::TensorBaseNode(int dim, DataType dtype)
    : dim(dim), dtype(dtype) {}

VType TensorBaseNode::getData(size_t offset) const { return data->at(offset); }

}; // namespace it