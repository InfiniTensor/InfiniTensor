#include "core/tensor.h"
#include "core/blob.h"
#include "core/graph.h"
#include "core/operator.h"
#include "core/runtime.h"
#include "utils/dataloader.h"
#include <cstring>
#include <limits>
#include <numeric>

namespace infini {

namespace {
size_t getTensorSize(const Shape &shape, DataType dtype) {
    size_t size = 1;
    for (const auto dim : shape) {
        IT_ASSERT(dim >= 0, "Tensor dimensions must be non-negative");
        if (dim == 0) {
            size = 0;
            continue;
        }
        IT_ASSERT(size <= std::numeric_limits<size_t>::max() /
                              static_cast<size_t>(dim),
                  "Tensor element count overflow");
        size *= static_cast<size_t>(dim);
    }
    IT_ASSERT(dtype.getSize() == 0 ||
                  size <= std::numeric_limits<size_t>::max() / dtype.getSize(),
              "Tensor byte size overflow");
    return size;
}
} // namespace

TensorObj::TensorObj(Shape shape_, DataType dtype, Runtime runtime)
    : TensorBaseObj(shape_.size(), dtype, runtime), shape(std::move(shape_)),
      _size(getTensorSize(shape, dtype)) {}

string TensorObj::toString() const {
    // Convert data pointer to string
    std::stringstream ss;
    if (data != nullptr)
        ss << data->getPtr<void *>();
    else
        ss << "nullptr data";
    string ret = "Tensor " + std::to_string(guid) + ", Fuid " +
                 std::to_string(fuid) + ", shape " + vecToString(shape) +
                 ", dtype " + dtype.toString() + ", " + runtime->toString() +
                 ", " + ss.str() + ", " + tensorTypeToString() + "\n";
    vector<UidBaseType> targetGuids;
    for (const auto &op : targets)
        targetGuids.emplace_back(op.lock()->getGuid());
    if (auto o = source.lock())
        ret += ", source " + std::to_string(o->getGuid());
    else
        ret += ", source None";
    ret += ", targets " + vecToString(targetGuids);
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

Shape TensorObj::getStride() const {
    Shape stride(getRank());
    ShapeElem p = 1;
    for (auto i = getRank(); i > 0; --i) {
        stride[i - 1] = p;
        p = p * shape[i - 1];
    }
    return stride;
}

void TensorObj::setShape(Shape shape_) {
    const auto size = getTensorSize(shape_, dtype);
    if (shape == shape_)
        return;
    shape = std::move(shape_);
    _size = size;
    notifyCaptureState(false);
}

void TensorObj::dumpData(std::ofstream &ofs) const {
    IT_ASSERT(data != nullptr);
    if (!runtime->isCpu())
        IT_TODO_HALT();

#define TRY_DUMP(N)                                                            \
    if (dtype == DataType(N))                                                  \
        ofs << dataToString<DT<N>::t>() << std::endl;

    TRY_DUMP(0)           // fmt: new line
    else TRY_DUMP(1)      //
        else TRY_DUMP(2)  //
        else TRY_DUMP(3)  //
        else TRY_DUMP(4)  //
        else TRY_DUMP(5)  //
        else TRY_DUMP(6)  //
        else TRY_DUMP(7)  //
        else TRY_DUMP(8)  //
        else TRY_DUMP(9)  //
        else TRY_DUMP(10) //
        else TRY_DUMP(11) //
        else TRY_DUMP(12) //
        else TRY_DUMP(13) //
        else TRY_DUMP(16) //
        else IT_TODO_HALT();
    ofs.flush();

#undef TRY_DUMP
}

void TensorObj::printData() const {
    IT_ASSERT(data != nullptr);
    if (!runtime->isCpu())
        IT_TODO_HALT();

#define TRY_PRINT(N)                                                           \
    if (dtype == DataType(N))                                                  \
        std::cout << dataToString<DT<N>::t>() << std::endl;

    TRY_PRINT(0)           // fmt: new line
    else TRY_PRINT(1)      //
        else TRY_PRINT(2)  //
        else TRY_PRINT(3)  //
        else TRY_PRINT(4)  //
        else TRY_PRINT(5)  //
        else TRY_PRINT(6)  //
        else TRY_PRINT(7)  //
        else TRY_PRINT(8)  //
        else TRY_PRINT(9)  //
        else TRY_PRINT(10) //
        else TRY_PRINT(11) //
        else TRY_PRINT(12) //
        else TRY_PRINT(13) //
        else TRY_PRINT(16) //
        else IT_TODO_HALT();

#undef TRY_PRINT
}

bool TensorObj::equalData(const Tensor &rhs, double relativeError) const {
    IT_ASSERT(data != nullptr);
    IT_ASSERT(rhs->data != nullptr);
    IT_ASSERT(getDType() == rhs->getDType());
    IT_ASSERT(runtime->isCpu());
    IT_ASSERT(rhs->getRuntime()->isCpu());
    if (size() != rhs->size())
        return false;

#define TEST_EQUAL(N)                                                          \
    if (dtype == DataType(N))                                                  \
        return equalDataImpl(getRawDataPtr<DT<N>::t *>(),                      \
                             rhs->getRawDataPtr<DT<N>::t *>(), size(),         \
                             relativeError);

    TEST_EQUAL(0)           // fmt: new line
    else TEST_EQUAL(1)      //
        else TEST_EQUAL(2)  //
        else TEST_EQUAL(3)  //
        else TEST_EQUAL(4)  //
        else TEST_EQUAL(5)  //
        else TEST_EQUAL(6)  //
        else TEST_EQUAL(7)  //
        else TEST_EQUAL(8)  //
        else TEST_EQUAL(9)  //
        else TEST_EQUAL(10) //
        else TEST_EQUAL(11) //
        else TEST_EQUAL(12) //
        else TEST_EQUAL(13) //
        else TEST_EQUAL(16) //
        else IT_TODO_HALT();

#undef TEST_EQUAL
}

void TensorObj::dataMalloc() {
    if (!data || data->getBytes() != getBytes()) {
        const bool releasedStorage = data != nullptr;
        data.reset();
        try {
            data = runtime->allocBlob(getBytes());
        } catch (...) {
            if (releasedStorage)
                notifyCaptureState(true);
            throw;
        }
        notifyCaptureState(true);
    }
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

void TensorObj::setDataBlob(const Blob &blob) {
    const auto previousStorageId = data ? data->getStorageId() : 0;
    const auto previousAddress = data ? data->getPtr<const void *>() : nullptr;
    const auto previousBytes = data ? data->getBytes() : 0;
    const auto nextStorageId = blob ? blob->getStorageId() : 0;
    const auto nextAddress = blob ? blob->getPtr<const void *>() : nullptr;
    const auto nextBytes = blob ? blob->getBytes() : 0;
    if (previousStorageId == nextStorageId && previousAddress == nextAddress &&
        previousBytes == nextBytes)
        return;
    data = blob;
    notifyCaptureState(previousStorageId != nextStorageId);
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
    for (auto i = 1; i < n; i++)
        offset = offset * dim.at(i) + pos.at(i);

    return offset;
}

size_t TensorObj::getOffsetByBroadcastOffset(size_t bcOffset,
                                             Shape bcDim) const {
    Shape bcPos = getPosByOffset(bcOffset, bcDim);

    Shape pos = bcPos;
    int n = shape.size();
    for (auto i = 0; i < n; i++)
        if (shape.at(i) == 1)
            pos[i] = 0;

    return getOffsetByPos(pos, shape);
}
}; // namespace infini
