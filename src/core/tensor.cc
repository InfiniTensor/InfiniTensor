#include <core/tensor.h>
namespace infini {

TensorNode::TensorNode(const Shape &shape, DataType dtype)
    : TensorBaseNode(shape.size(), dtype), shape(shape) {}

void TensorNode::dataMalloc() {
    IT_ASSERT(data == nullptr);
    // initialized to zero
    data.reset(reinterpret_cast<VType *>(calloc(size(), sizeof(VType))));
}

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

size_t TensorNode::size() const {
    size_t ret = 1;
    for (const auto &d : shape)
        ret *= d;
    return ret;
}

void TensorNode::copyData(VType *dptr) {
    IT_ASSERT(data != nullptr);
    size_t sz = size();
#pragma omp parallel for
    for (size_t i = 0; i < sz; ++i) {
        data[i] = dptr[i];
    }
}

void TensorNode::printData() const {
    IT_ASSERT(data != nullptr);
    std::cout << "Tensor: " << guid << std::endl;
    auto numDims = shape.size();
    auto dimSzVec = std::vector<int>(numDims, 1);
    dimSzVec[numDims - 1] = shape[numDims - 1];
    for (int i = numDims - 1; i != 0; --i)
        dimSzVec[i - 1] = dimSzVec[i] * shape[i - 1];
    for (size_t i = 0, iEnd = size(); i < iEnd; ++i) {
        for (size_t j = 0; j < numDims; ++j) {
            if (i % dimSzVec[j] == 0) {
                std::cout << "[";
            }
        }
        std::cout << data[i];
        for (size_t j = 0; j < numDims; ++j) {
            if ((int)i % dimSzVec[j] == dimSzVec[j] - 1) {
                std::cout << "]";
            }
        }
        if (i != size() - 1)
            std::cout << ", ";
        if ((int)i % dimSzVec[numDims - 1] == dimSzVec[numDims - 1] - 1)
            std::cout << std::endl;
    }
}

bool TensorNode::equalData(const Tensor &rhs) const {
    IT_ASSERT(data != nullptr);
    IT_ASSERT(rhs->data != nullptr);
    if (shape != rhs->getDims())
        return false;
    size_t sz = size();
    for (size_t i = 0; i < sz; ++i)
        if (data[i] != rhs->data[i])
            return false;
    return true;
}

}; // namespace infini