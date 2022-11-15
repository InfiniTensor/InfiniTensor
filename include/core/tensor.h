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
    Fuid fuid; // Tensor cloned from a common tensor share the same id. Tensors
               // constructed from common constructor has a new id.

  public:
    TensorObj(const Shape &shape, DataType dtype, Runtime runtime);
    virtual ~TensorObj() {}
    string toString() const override;

    size_t size() const;
    size_t getBytes() const;

    Shape getDims() const { return shape; }
    vector<size_t> getStride() const;
    size_t getOffset(const Shape &ds) const;
    using TensorBaseObj::getData;
    VType getData(const Shape &pos) const;
    void dataMalloc();
    GuidBaseType getFuid() const { return fuid; }

    void load(std::string file_path);
    void save(std::string file_path);

    template <typename T> void copyData(const T *dptr) {
        IT_ASSERT(DataType::get<T>() == dtype);
        IT_ASSERT(data != nullptr);
        runtime->copyBlobFromCPU(getRawDataPtr<void *>(), dptr, getBytes());
    }

    template <typename T> void copyData(vector<T> dataVector) {
        IT_ASSERT(DataType::get<T>() == dtype);
        IT_ASSERT(dataVector.size() >= size());
        copyData(dataVector.data());
    }

    void copyData(const TensorObj *src);
    void copyData(const Tensor &src) { copyData(src.get()); }
    void setData(
        const std::function<void(void *, size_t, DataType)> &generator) const {
        IT_ASSERT(data != nullptr);
        if (!runtime->isCpu()) {
            IT_TODO_HALT();
        }
        generator(data->getPtr<void *>(), size(), dtype);
    }
    Tensor clone() const {
        auto obj = make_ref<TensorObj>(*this);
        obj->freeData();
        obj->inputOf.clear();
        obj->outputOf.reset();
        return obj;
    }
    Tensor clone(Runtime runtime) const {
        // TODO: use copy constructor
        auto obj = make_ref<TensorObj>(shape, dtype, runtime);
        obj->dataMalloc();
        obj->copyData(this);
        return obj;
    }

    void printData() const;
    bool equalData(const Tensor &rhs) const;

    template <typename T> bool equalData(const vector<T> &dataVector) {
        IT_ASSERT(DataType::get<T>() == dtype);
        IT_ASSERT(size() == dataVector.size());
        return equalDataImpl(getRawDataPtr<T *>(), dataVector.data(), size());
    }

    size_t getOffsetByBroadcastOffset(size_t bcOffset, Shape bcShape) const;

  private:
    void printDataFloat() const;
    void printDataUint32_t() const;

    template <typename T>
    bool equalDataImpl(const T *a, const T *b, size_t size) const {
        for (size_t i = 0; i < size; ++i) {
            if constexpr (std::is_integral_v<T>) {
                if (a[i] != b[i])
                    return false;
            } else if constexpr (std::is_floating_point_v<T>) {
                if (fabs(a[i] - b[i]) / std::max(fabs(a[i]), fabs(b[i])) >
                    1e-6) {
                    printf("Error on %lu: %f %f\n", i, a[i], b[i]);
                    return false;
                }
            } else
                static_assert(!sizeof(T), "Unsupported data type");
        }
        return true;
    }

    Shape getPosByOffset(size_t offset, Shape dim) const;
    size_t getOffsetByPos(Shape pos, Shape dim) const;

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
