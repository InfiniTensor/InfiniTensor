#pragma once
#include "core/tensor_base.h"
#include <cmath>

namespace infini {

// TODO: how to deal with this
using ShapeElem = int;
using Shape = vector<ShapeElem>;
class TensorObj : public TensorBaseObj {
  private:
    Shape shape;

  public:
    TensorObj(const Shape &shape, DataType dtype);
    virtual ~TensorObj() {}
    string toString() const override;

    size_t size() const;

    Shape getDims() const { return shape; }

    size_t getOffset(const Shape &ds) const;
    using TensorBaseObj::getData;
    VType getData(const Shape &pos) const;
    void dataMalloc(const Runtime &runtime);
    // void copyData(VType *dptr);
    template <typename T> void copyData(const T *dptr);
    void copyData(vector<VType> dataVector);
    void copyData(vector<float> dataVector);
    void printData() const;
    // TODO: merge these methods
    bool equalData(const Tensor &rhs) const;
    template <typename T> bool equalData(const Tensor &rhs) const {
        IT_ASSERT(data != nullptr);
        IT_ASSERT(rhs->data != nullptr);
        // TODO: deal with data type
        auto ptr = data->getPtr<T *>();
        auto ptrRhs = rhs->data->getPtr<T *>();
        if (shape != rhs->getDims())
            return false;
        size_t sz = size();
        for (size_t i = 0; i < sz; ++i)
            if (fabs(ptr[i] - ptrRhs[i]) /
                    std::max(fabs(ptr[i]), fabs(ptrRhs[i])) >
                1e-6) {
                printf("Error on %lu: %f %f\n", i, ptr[i], ptrRhs[i]);
                return false;
            }
        return true;
    }
    void setData(
        const std::function<void(void *, size_t, DataType)> &generator) const {
        generator(data->getPtr<void *>(), size(), dtype);
    }

  private:
    void printDataFloat() const;
    void printDataUint32_t() const;
    // void setDims(const Dim &dms) { dims = dms; }

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