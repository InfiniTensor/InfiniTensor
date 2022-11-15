#pragma once
#include "core/blob.h"
#include "core/data_type.h"
#include "core/object.h"
#include "core/runtime.h"

namespace infini {

class TensorBaseObj : public Object {
  public:
    // enum TensorType {
    //     Input,
    //     Weight,
    //     Invalid,
    //     NotCounted,
    // };

  protected:
    int dim;

    DataType dtype;
    vector<WRef<OperatorObj>> inputOf;
    WRef<OperatorObj> outputOf;
    Blob data;
    Runtime runtime;

  public:
    TensorBaseObj(int dim, DataType dtype, Runtime runtime);
    virtual ~TensorBaseObj() {}

    void dataMalloc(const Blob &blob) {
        IT_ASSERT(data == nullptr);
        data = blob;
    }
    Blob getDataBlob() const { return data; }
    void freeData() { data = nullptr; }
    template <typename T> T getRawDataPtr() const {
        static_assert(std::is_pointer_v<T>,
                      "Raw data pointer has a type of pointer");
        IT_ASSERT(data != nullptr);
        return data->getPtr<T>();
    }
    VType getData(size_t offset) const;

    DataType getDType() const { return dtype; }
    Runtime getRuntime() const { return runtime; }

    void addInputOf(const Operator &op) { inputOf.emplace_back(op); }
    void setOutputOf(const Operator &op) { outputOf = op; }
    OpVec getInputOf() { return wrefs_to_refs(inputOf); }
    Operator getOutputOf() { return outputOf.lock(); }
    //     std::pair<Operator *, int> getOutputOfWithIndex();

    //     bool setScalar(VType val) {
    //         if (data == nullptr || !dims.empty())
    //             return false;
    //         data[0] = val;
    //         return true;
    //     }

    //     bool setData(const Dim &ds, VType val) {
    //         if (data == nullptr || ds.size() != dims.size())
    //             return false;
    //         data[getOffset(ds)] = val;
    //         return true;
    //     }

    //     bool setData(size_t pos, VType val) {
    //         if (data == nullptr || pos >= size())
    //             return false;
    //         data[pos] = val;
    //         return true;
    //     }

    //     VType getScalar() { return data == nullptr ? 0 : data[0]; }

    //     VType getBroadcastData(const Dim &ds) {
    //         assert(data != nullptr);
    //         auto offset = getBroadcastOffset(ds);
    //         return offset == (size_t)-1 ? 0 : data[getOffset(ds)];
    //     }

    //     VType getBroadcastData(size_t pos) {
    //         assert(data != nullptr);
    //         return data[pos % size()];
    //     }

    //     size_t getBroadcastOffset(const Dim &ds) {
    //         assert(ds.size() >= dims.size());
    //         auto nDim = dims.size();
    //         auto nBroadcastDim = ds.size() - nDim;
    //         for (size_t i = 0; i < nDim; ++i)
    //             if (ds[nBroadcastDim + i] < 0 || ds[nBroadcastDim + i] >=
    //             dims[i])
    //                 return (size_t)-1;
    //         size_t idx = 0;
    //         for (size_t i = 0; i < nDim; ++i)
    //             idx = idx * dims[i] + ds[nBroadcastDim + i];
    //         return idx;
    //     }
};

} // namespace infini
