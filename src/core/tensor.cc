#include "core/tensor.h"
#include "core/blob.h"
#include "core/operator.h"
#include "core/runtime.h"
#include "utils/dataloader.h"
#include <cstring>
#include <numeric>

namespace infini {

TensorObj::TensorObj(Shape shape_, DataType dtype, Runtime runtime)
    : TensorBaseObj(shape.size(), dtype, runtime), shape(std::move(shape_)),
      _size(shape.empty()
                ? 0
                : std::accumulate(shape.begin(), shape.end(), 1,
                                  [](auto acc, auto x) { return acc * x; })) {}

string TensorObj::toString() const {
    // Convert data pointer to string
    std::stringstream ss;
    if (data != nullptr)
        ss << data->getPtr<void *>();
    else
        ss << "nullptr data";
    string ret = "Tensor " + std::to_string(guid) + ", Fuid " +
                 std::to_string(fuid) + ", shape " + vecToString(shape) +
                 ", dtype " + dtype.toString();
    vector<UidBaseType> targetGuids;
    for (const auto &op : targets)
        targetGuids.emplace_back(op.lock()->getGuid());
    if (auto o = source.lock())
        ret += ", source " + std::to_string(o->getGuid());
    else
        ret += ", source None";
    ret += ", targets " + vecToString(targetGuids);
    ret += ", " + runtime->toString() + ", " + ss.str();
    return ret;
}

size_t TensorObj::getOffset(const vector<int> &pos) const {
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

vector<size_t> TensorObj::getStride() const {
    vector<size_t> ret;
    size_t stride = 1;
    for (int i = shape.size() - 1; i >= 1; i--) {
        ret.emplace(ret.begin(), stride);
        stride *= shape.at(i);
    }
    ret.emplace(ret.begin(), stride);
    return ret;
}

void TensorObj::printData() const {
    IT_ASSERT(data != nullptr);
    void *ptr = nullptr;
    Blob buffer;
    if (!runtime->isCpu()) {
        buffer = NativeCpuRuntimeObj::getInstance()->allocBlob(getBytes());
        runtime->copyBlobToCPU(buffer->getPtr<void *>(),
                               getRawDataPtr<void *>(), getBytes());
        ptr = buffer->getPtr<void *>();
    } else
        ptr = data->getPtr<float *>();
    if (dtype == DataType::Float32)
        printDataFloat(static_cast<float *>(ptr));
    else if (dtype == DataType::UInt32)
        printDataUint32_t(static_cast<uint32_t *>(ptr));
    else
        IT_TODO_HALT();
}

void TensorObj::printDataFloat(float *ptr) const {
    std::cout << "Tensor: " << guid << std::endl;
    auto numDims = shape.size();
    auto dimSzVec = std::vector<int>(numDims, 1);
    dimSzVec[numDims - 1] = shape[numDims - 1];
    for (int i = numDims - 1; i != 0; --i)
        dimSzVec[i - 1] = dimSzVec[i] * shape[i - 1];
    for (size_t i = 0, iEnd = size(); i < iEnd; ++i) {
        if (iEnd > 1000 && i > 20 && i < iEnd - 20) {
            printf("... , ");
            i = iEnd - 20;
            continue;
        }
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

void TensorObj::printDataUint32_t(uint32_t *ptr) const {
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

bool TensorObj::equalData(const Tensor &rhs, double relativeError) const {
    IT_ASSERT(data != nullptr);
    IT_ASSERT(rhs->data != nullptr);
    IT_ASSERT(getDType() == rhs->getDType());
    IT_ASSERT(runtime->isCpu());
    IT_ASSERT(rhs->getRuntime()->isCpu());
    if (size() != rhs->size())
        return false;
    if (getDType() == DataType::UInt32)
        return equalDataImpl(getRawDataPtr<uint32_t *>(),
                             rhs->getRawDataPtr<uint32_t *>(), size(), 0);
    else if (getDType() == DataType::Float32)
        return equalDataImpl(getRawDataPtr<float *>(),
                             rhs->getRawDataPtr<float *>(), size(),
                             relativeError);
    else
        IT_TODO_HALT();
}

void TensorObj::dataMalloc() {
    if (data == nullptr)
        data = runtime->allocBlob(getBytes());
}

void TensorObj::copyData(const TensorObj *src) {
    IT_ASSERT(dtype == src->getDType());
    IT_ASSERT(size() == src->size());
    runtime->copyBlob(this, src);
}

void TensorObj::setData(
    const std::function<void(void *, size_t, DataType)> &generator) const {
    IT_ASSERT(data != nullptr);
    if (runtime->isCpu()) {
        generator(getRawDataPtr<void *>(), size(), dtype);
    } else {
        // Create a CPU buffer for the generetor and copy results to the device
        auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
        size_t nBytes = size() * dtype.getSize();
        Blob buffer = cpuRuntime->allocBlob(nBytes);
        generator(buffer->getPtr<void *>(), size(), dtype);
        runtime->copyBlobFromCPU(getRawDataPtr<void *>(),
                                 buffer->getPtr<void *>(), nBytes);
    }
}

void TensorObj::load(std::string file_path) { loadTensorData(this, file_path); }

void TensorObj::save(std::string file_path) { saveTensorData(this, file_path); }

Shape TensorObj::getPosByOffset(size_t offset, Shape dim) const {
    Shape pos = dim;
    for (int i = dim.size() - 1; i >= 0; i--) {
        pos[i] = offset % dim.at(i);
        offset = (offset - pos[i]) / dim.at(i);
    }
    return pos;
}

size_t TensorObj::getOffsetByPos(Shape pos, Shape dim) const {
    int n = dim.size();
    size_t offset = pos.at(0);
    for (auto i = 1; i < n; i++) {
        offset = offset * dim.at(i) + pos.at(i);
    }
    return offset;
}

size_t TensorObj::getOffsetByBroadcastOffset(size_t bcOffset,
                                             Shape bcDim) const {
    Shape bcPos = getPosByOffset(bcOffset, bcDim);

    Shape pos = bcPos;
    int n = shape.size();
    for (auto i = 0; i < n; i++) {
        if (shape.at(i) == 1)
            pos[i] = 0;
    }
    return getOffsetByPos(pos, shape);
}
}; // namespace infini
