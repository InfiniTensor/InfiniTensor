#include "core/tensor.h"
#include "core/blob.h"
#include "core/runtime.h"

namespace infini {

TensorObj::TensorObj(const Shape &shape, DataType dtype, Runtime runtime)
    : TensorBaseObj(shape.size(), dtype, runtime), shape(shape) {}

VType TensorObj::getData(const Shape &pos) const {
    return getData(getOffset(pos));
}

string TensorObj::toString() const { return "Tensor " + std::to_string(guid); }

size_t TensorObj::getOffset(const Shape &pos) const {
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

size_t TensorObj::size() const {
    size_t ret = 1;
    for (const auto &d : shape)
        ret *= d;
    return ret;
}

size_t TensorObj::getBytes() const { return size() * dtype.getSize(); }

void TensorObj::printData() const {
    IT_ASSERT(data != nullptr);
    if (!runtime->isCpu())
        IT_TODO_HALT();
    if (dtype == DataType::Float32)
        printDataFloat();
    else if (dtype == DataType::UInt32)
        printDataUint32_t();
    else
        IT_TODO_HALT();
}

void TensorObj::printDataFloat() const {
    std::cout << "Tensor: " << guid << std::endl;
    auto numDims = shape.size();
    auto dimSzVec = std::vector<int>(numDims, 1);
    auto ptr = data->getPtr<float *>();
    dimSzVec[numDims - 1] = shape[numDims - 1];
    for (int i = numDims - 1; i != 0; --i)
        dimSzVec[i - 1] = dimSzVec[i] * shape[i - 1];
    for (size_t i = 0, iEnd = size(); i < iEnd; ++i) {
        for (size_t j = 0; j < numDims; ++j) {
            if (i % dimSzVec[j] == 0) {
                std::cout << "[";
            }
        }
        printf("%.1f", ptr[i]);
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

void TensorObj::printDataUint32_t() const {
    IT_ASSERT(data != nullptr);
    std::cout << "Tensor: " << guid << std::endl;
    auto numDims = shape.size();
    auto dimSzVec = std::vector<int>(numDims, 1);
    auto ptr = data->getPtr<VType *>();
    dimSzVec[numDims - 1] = shape[numDims - 1];
    for (int i = numDims - 1; i != 0; --i)
        dimSzVec[i - 1] = dimSzVec[i] * shape[i - 1];
    for (size_t i = 0, iEnd = size(); i < iEnd; ++i) {
        for (size_t j = 0; j < numDims; ++j) {
            if (i % dimSzVec[j] == 0) {
                std::cout << "[";
            }
        }
        std::cout << ptr[i];
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

template <typename T> bool TensorObj::equalDataInt(const Tensor &rhs) const {
    auto ptr = data->getPtr<uint32_t *>();
    auto ptrRhs = rhs->data->getPtr<uint32_t *>();
    if (shape != rhs->getDims())
        return false;
    size_t sz = size();
    for (size_t i = 0; i < sz; ++i)
        if (ptr[i] != ptrRhs[i])
            return false;
    return true;
}

template <typename T> bool TensorObj::equalDataFloat(const Tensor &rhs) const {
    IT_ASSERT(data != nullptr);
    IT_ASSERT(rhs->data != nullptr);
    // TODO: deal with data type
    auto ptr = data->getPtr<T *>();
    auto ptrRhs = rhs->data->getPtr<T *>();
    if (shape != rhs->getDims())
        return false;
    size_t sz = size();
    for (size_t i = 0; i < sz; ++i)
        if (fabs(ptr[i] - ptrRhs[i]) / std::max(fabs(ptr[i]), fabs(ptrRhs[i])) >
            1e-6) {
            printf("Error on %lu: %f %f\n", i, ptr[i], ptrRhs[i]);
            return false;
        }
    return true;
}

bool TensorObj::equalData(const Tensor &rhs) const {
    IT_ASSERT(data != nullptr);
    IT_ASSERT(rhs->data != nullptr);
    IT_ASSERT(getDType() == rhs->getDType());
    if (getDType() == DataType::UInt32)
        return equalDataInt<uint32_t>(rhs);
    else if (getDType() == DataType::Float32)
        return equalDataInt<float>(rhs);
    else
        IT_TODO_HALT();
}

void TensorObj::dataMalloc(const Runtime &runtime) {
    IT_ASSERT(data == nullptr);
    size_t bytesPerElement;
    if (getDType() == DataType::Float32)
        bytesPerElement = sizeof(float);
    else if (getDType() == DataType::UInt32)
        bytesPerElement = sizeof(uint32_t);
    data = runtime->allocBlob(size() * bytesPerElement);
}

}; // namespace infini