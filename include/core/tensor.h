#pragma once
#include "core/tensor_base.h"
#include "core/tensor_type.h"
#include "utils/data_convert.h"
#include <cmath>
#include <cstring>
#include <fstream>

namespace infini {
namespace ops {
class Tensor;
}

// TODO: how to deal with this
using ShapeElem = int;
using Shape = vector<ShapeElem>;
class TensorObj : public TensorBaseObj {
  private:
    Shape shape;
    size_t _size; // Cache of Π(shape).
    Fuid fuid;    // Cloned tensors share the same id. Tensors constructed from
                  // scratch have a new id.
    TensorType tensorType = TensorType::others;

  public:
    TensorObj(Shape shape, DataType dtype, Runtime runtime);
    virtual ~TensorObj() {}
    string toString() const override;

    size_t size() const { return _size; }
    size_t getBytes() const { return _size * dtype.getSize(); }

    Shape getDims() const { return shape; }
    void setShape(Shape shape_);
    size_t getRank() const { return shape.size(); }
    Shape getStride() const;
    size_t getOffset(const vector<int> &ds) const;
    void dataMalloc();
    UidBaseType getFuid() const { return fuid; }
    bool isWeight() const { return tensorType == TensorType::weight; }
    bool isInput() const { return tensorType == TensorType::input; }
    bool isOutput() const { return tensorType == TensorType::output; }
    bool isOthers() const { return tensorType == TensorType::others; }
    void setWeight() { tensorType = TensorType::weight; }
    void setInput() {
        if (!this->isWeight()) {
            tensorType = TensorType::input;
        }
    }
    void setOutput() {
        if (!this->isWeight()) {
            tensorType = TensorType::output;
        }
    }
    string tensorTypeToString() const {
        switch (tensorType) {
        case TensorType::weight:
            return "weight";
            break;
        case TensorType::input:
            return "input";
            break;
        case TensorType::output:
            return "output";
            break;
        case TensorType::others:
            return "others";
            break;

        default:
            return "unknown tensor type";
            break;
        }
    }

    void load(std::string file_path);
    void save(std::string file_path);

    void copyin(const void *ptr, size_t size) {
        runtime->copyBlobFromCPU(getRawDataPtr<void *>(), ptr, size);
    }
    void copyout(void *ptr, size_t size) const {
        runtime->copyBlobToCPU(ptr, getRawDataPtr<void *>(), size);
    }

    // Copy elements from `data`.
    template <typename T> void copyin(const vector<T> &data) {
        IT_ASSERT(DataType::get<T>() == dtype.cpuTypeInt());
        IT_ASSERT(data.size() == _size);
        copyin(data.data(), getBytes());
    }
    // Copy all the elements to a vector.
    template <typename T> auto copyout() const {
        IT_ASSERT(DataType::get<T>() == dtype.cpuTypeInt());
        std::vector<T> ans(_size);
        copyout(ans.data(), getBytes());
        return ans;
    }
    // Copy the element at `pos`.
    template <typename T> auto copyOne(const vector<int> &pos) const {
        IT_ASSERT(DataType::get<T>() == dtype.cpuTypeInt());
        auto offset = getOffset(pos);
        auto bytes = dtype.getSize();
        T ans;
        runtime->copyBlobToCPU(
            &ans, getRawDataPtr<uint8_t *>() + offset * bytes, bytes);
        return ans;
    }

    void copyData(const TensorObj *src);
    void copyData(const Tensor &src) { copyData(src.get()); }

    // TODO: Rename this function later, because it is confused that it will
    // change the field data, but actually it generates data and maybe copy to
    // device.
    // FIXME: std::fucntion copies the generator instead of passing it by ref.
    // Thus the internal state of generator cannot be updated.
    void setData(
        std::function<void(void *, size_t, DataType)> const &generator) const;

    void setDataBlob(const Blob &blob);

    Tensor clone() const {
        auto obj = make_ref<TensorObj>(*this);
        obj->freeData();
        obj->targets.clear();
        obj->source.reset();
        return obj;
    }
    Tensor clone(Runtime runtime) const {
        auto obj = make_ref<TensorObj>(*this);
        obj->runtime = runtime;
        obj->freeData();
        obj->targets.clear();
        obj->source.reset();
        if (hasData()) {
            obj->dataMalloc();
            obj->copyData(this);
        }
        return obj;
    }

    void printData() const;
    void dumpData(std::ofstream &ofs) const;
    bool equalData(const Tensor &rhs, double relativeError = 1e-6) const;

    template <typename T> bool equalData(const vector<T> &dataVector) {
        IT_ASSERT(size() == dataVector.size());
        if (dtype == DataType::Float16) {
            return equalDataImpl_fp16(getRawDataPtr<uint16_t *>(),
                                      (float *)dataVector.data(), size());
        }
        IT_ASSERT(DataType::get<T>() == dtype.cpuTypeInt());
        return equalDataImpl(getRawDataPtr<T *>(), dataVector.data(), size());
    }

    size_t getOffsetByBroadcastOffset(size_t bcOffset, Shape bcShape) const;

  private:
    template <class T> string dataToString() const {
        std::stringstream builder;
        builder << "Tensor: " << guid << std::endl;

        auto numDims = shape.size();
        auto dimSzVec = vector<int>(numDims, 1);
        auto ptr = data->getPtr<T *>();
        dimSzVec[numDims - 1] = shape[numDims - 1];

        for (int i = numDims - 1; i != 0; --i)
            dimSzVec[i - 1] = dimSzVec[i] * shape[i - 1];

        for (size_t i = 0, iEnd = size(); i < iEnd; ++i) {
            for (size_t j = 0; j < numDims; ++j)
                if (i % dimSzVec[j] == 0)
                    builder << "[";

            builder << ptr[i];
            for (size_t j = 0; j < numDims; ++j)
                if ((int)i % dimSzVec[j] == dimSzVec[j] - 1)
                    builder << "]";

            if (i != size() - 1)
                builder << ", ";

            auto column = (size_t)dimSzVec[numDims - 1];
            if (i % column == column - 1)
                builder << std::endl;
        }
        return builder.str();
    }

    template <typename T>
    bool equalDataImpl(const T *a, const T *b, size_t size,
                       double relativeError = 1e-6) const {
        for (size_t i = 0; i < size; ++i) {
            if constexpr (std::is_integral_v<T>) {
                if (a[i] != b[i])
                    return false;
            } else if constexpr (std::is_floating_point_v<T>) {
                if (std::min(fabs(a[i]), fabs(b[i])) == 0. &&
                    fabs(a[i] - b[i]) > relativeError) {
                    printf("Error on %lu: %f %f\n", i, a[i], b[i]);
                    return false;
                } else if (std::min(fabs(a[i]), fabs(b[i])) != 0. &&
                           fabs(a[i] - b[i]) /
                                   std::max(fabs(a[i]), fabs(b[i])) >
                               relativeError) {
                    printf("Error on %lu: %f %f\n", i, a[i], b[i]);
                    return false;
                }
            } else {
                static_assert(!sizeof(T), "Unsupported data type");
            }
        }
        return true;
    }

    bool equalDataImpl_fp16(const uint16_t *a, const float *b,
                            size_t size) const {
        for (size_t i = 0; i < size; ++i) {
            auto a_fp32 = fp16_to_float(a[i]);
            auto b_fp32 = b[i];
            if (fabs(a_fp32 - b_fp32) / std::max(fabs(a_fp32), fabs(b_fp32)) >
                1e-6) {
                printf("Error on %lu: %f %f\n", i, a_fp32, b_fp32);
                return false;
            }
        }
        return true;
    }

    Shape getPosByOffset(size_t offset, Shape dim) const;
    size_t getOffsetByPos(Shape pos, Shape dim) const;
};
infini::ops::Tensor toInfiniOpsTensor(const TensorObj *tensor);

} // namespace infini
