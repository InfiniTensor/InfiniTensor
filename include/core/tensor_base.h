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
    vector<WRef<TensorBaseObj>> inputOf;
    WRef<TensorBaseObj> outputOf;
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
    template <typename T> T getRawDataPtr() const {
        static_assert(std::is_pointer_v<T>,
                      "Raw data pointer has a type of pointer");
        IT_ASSERT(data != nullptr);
        return data->getPtr<T>();
    }
    VType getData(size_t offset) const;

    DataType getDType() const { return dtype; }
    Runtime getRuntime() const { return runtime; }

    // uint64_t getHash() const { return hash; }

    //     void setInputOf(const OpVec &ops) {
    //         inputOf.clear();
    //         for (const auto &op : ops)
    //             inputOf.emplace_back(op);
    //     }
    //     void addInputOf(Operator op) { inputOf.emplace_back(op); }
    //     void setOutputOf(Operator op) { outputOf = op; }

    //     const OpVec &getInputOf() { return inputOf; }
    //     Operator *getOutputOf() { return outputOf; }
    //     std::pair<Operator *, int> getOutputOfWithIndex();

    //     const Dim &getDims() const { return dims; }
    //     void setDims(const Dim &dms) { dims = dms; }

    //     bool dataRand(int seed = 0) {
    //         if (data == nullptr)
    //             data = new VType[size()];
    //         if (!random_inited)
    //             initFastrand();
    //         // srand(seed);
    //         // faster rand generator; parallel
    //         size_t iEnd = size();
    //         // std::cerr << "Init beginned " << std::endl;
    // #pragma omp parallel for
    //         for (size_t i = 0; i < iEnd; ++i)
    //             data[i] = fastrand(random_seed[omp_get_thread_num() * 16]) %
    //             10000;
    //         // std::cerr << "Init finished" << std::endl;
    //         computed = ComputedFull;
    //         return true;
    //     }

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

    //     VType getData(const Dim &ds) {
    //         assert(data != nullptr);
    //         auto offset = getOffset(ds);
    //         return offset == (size_t)-1 ? 0 : data[getOffset(ds)];
    //     }

    //     VType getData(size_t pos) {
    //         assert(data != nullptr);
    //         assert(pos < size());
    //         return data[pos];
    //     }

    //     VType *getDataPtr() const { return data; }

    //     size_t getOffset(const Dim &ds) {
    //         auto nDim = ds.size();
    //         assert(dims.size() == nDim);
    //         if (ds.empty())
    //             return 0;
    //         for (size_t i = 0; i < nDim; ++i)
    //             if (ds[i] < 0 || ds[i] >= dims[i])
    //                 return (size_t)-1;
    //         size_t idx = ds[0];
    //         size_t dm = 0;
    //         while (++dm < nDim)
    //             idx = idx * dims[dm] + ds[dm];
    //         return idx;
    //     }

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

    //     void itInit() { it = Dim(dims.size(), 0); }

    //     void itReset() {
    //         itInit();
    //         for (size_t i = 0, iEnd = it.size(); i < iEnd; ++i)
    //             it[i] = 0;
    //     }

    //     bool itValid() {
    //         if (it.size() != dims.size())
    //             return false;
    //         for (size_t i = 0, iEnd = it.size(); i < iEnd; ++i)
    //             if (it[i] >= dims[i])
    //                 return false;
    //         return true;
    //     }

    //     const Dim &itGet() { return it; }

    //     void itNext() {
    //         auto p = it.size() - 1;
    //         it[p] += 1;
    //         while (p >= 1) {
    //             if (it[p] == dims[p]) {
    //                 it[p] = 0;
    //                 it[--p] += 1;
    //             } else
    //                 break;
    //         }
    //     }

    //     size_t size() const {
    //         size_t sz = 1;
    //         auto dm = dims.size();
    //         while (dm > 0)
    //             sz *= dims[--dm];
    //         return sz;
    //     }

    //     TensorType getType() const { return type; }
    //     void setType(TensorType ty) { type = ty; }

    //     static inline void initFastrand() {
    //         assert(omp_get_max_threads() <= 256);
    //         // srand(0); // constant seed for test
    //         // align random_seed to avoid false sharing
    //         for (int i = 0; i < 256 * 16; ++i) {
    //             // random_seed[i] = rand();
    //             // constant random seed for test
    //             random_seed[i] = i;
    //         }
    //         random_inited = true;
    //     }

    //     static inline int fastrand(int &g_seed) {
    //         g_seed = (214013 * g_seed + 2531011);
    //         return (g_seed >> 16) & 0x7FFF;
    //     }

    //     std::vector<std::vector<int>> const *getSplittingPoints() const {
    //         assert(!splittingPoints.empty());
    //         return &splittingPoints;
    //     }

    //     bool setSplittingPoints(std::vector<std::vector<int>> value) {
    //         assert(!value.empty());
    //         splittingPoints = value;
    //         return true;
    //     }

    //     void printSplittingPoints() {
    //         if (splittingPoints.empty())
    //             printf("Empty SplittingPoints");
    //         else {
    //             printf("[");
    //             for (auto &vs : splittingPoints) {
    //                 printf("[");
    //                 for (auto v : vs)
    //                     printf("%2d,", v);
    //                 printf("],");
    //             }
    //             printf("]");
    //         }
    //     }

    //     void initSplittingPoints() {
    //     splittingPoints.resize(getDims().size()); }

    //     void printShape();
};

} // namespace infini
