#pragma once
#include "core/blob.h"
#include "core/data_type.h"
#include "core/object.h"
#include "core/runtime.h"
namespace infini {
class GraphObj;
class TensorBaseObj : public Object {
    friend class GraphObj;

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
    vector<WRef<OperatorObj>> targets;
    WRef<OperatorObj> source;
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
    bool hasData() const { return data != nullptr; }
    void freeData() { data = nullptr; }
    template <typename T> T getRawDataPtr() const {
        static_assert(std::is_pointer_v<T>,
                      "Raw data pointer has a type of pointer");
        IT_ASSERT(data != nullptr);
        return data->getPtr<T>();
    }

    DataType getDType() const { return dtype; }
    Runtime getRuntime() const { return runtime; }

    //     std::pair<Operator *, int> getOutputOfWithIndex();

    bool hasTarget() const { return !targets.empty(); }

    OpVec getTargets() const { return wrefs_to_refs(targets); }
    Operator getSource() const { return source.lock(); }

  private:
    void addTarget(const Operator &op) { targets.emplace_back(op); }
    void setSource(const Operator &op) { source = op; }
    void removeTarget(const Operator &op) {
        for (auto itr = targets.begin(); itr != targets.end();) {
            if (itr->lock() == op)
                itr = targets.erase(itr);
            else
                ++itr;
        }
    }
    //     std::pair<Operator *, int> getSourceWithIndex();

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
