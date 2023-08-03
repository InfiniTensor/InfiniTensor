#include "core/tensor.h"
#include "core/blob.h"
#include "core/operator.h"
#include "core/runtime.h"
#include "utils/dataloader.h"
#include <cstring>
#include <numeric>

namespace infini {

TensorObj::TensorObj(Shape shape_, DataType dtype, Runtime runtime)
    : TensorBaseObj(shape_.size(), dtype, runtime), shape(std::move(shape_)),
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
                 ", dtype " + dtype.toString() + ", " + runtime->toString() +
                 ", " + ss.str() + "\n";
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
                             rhs->getRawDataPtr<DT<N>::t *>(), size());

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
        else IT_TODO_HALT();

#undef TEST_EQUAL
}

void TensorObj::dataMalloc() {
    if (!data)
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

void TensorObj::setDataBlob(const Blob &blob) { this->data = blob; }

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
