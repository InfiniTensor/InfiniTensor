#include <core/tensor.h>
namespace it {

TensorNode::TensorNode(const Shape &shape, DataType dtype)
    : TensorBaseNode(shape.size(), dtype), shape(shape) {}

VType TensorNode::getData(const Shape &pos) const {
    return getData(getOffset(pos));
}

string TensorNode::toString() const {
    return "TensorNode " + std::to_string(guid);
}

size_t TensorNode::getOffset(const Shape &pos) const {
    auto nDim = pos.size();
    IT_ASSERT(shape.size() == nDim);
    if (pos.empty())
        return 0;
    for (size_t i = 0; i < nDim; ++i)
        IT_ASSERT(pos[i] < 0 || pos[i] >= shape[i]);
    size_t idx = pos[0];
    size_t dm = 0;
    while (++dm < nDim)
        idx = idx * shape[dm] + pos[dm];
    return idx;
}

}; // namespace it