#pragma once
#include "core/tensor_base.h"
#include "core/tensor_type.h"
#include "utils/data_convert.h"
#include <cmath>
#include <cstring>
#include <fstream>

#if USE_CUDA
#include "cuda/cuda_runtime.h"
#endif
#if USE_BANG
#include "bang/bang_runtime.h"
#endif
namespace infini {

// TODO: how to deal with this
using ShapeElem = int;
using Shape = vector<ShapeElem>;

/// @brief Describes whether one dimension may change between executions.
///
/// It mirrors how ONNX declares a dimension: `dim_value` yields a fixed
/// dimension, `dim_param` yields a symbolic one carrying its name, and a
/// dimension with neither is dynamic but anonymous.
struct DimDesc {
    bool dynamic = false; ///< Whether the dimension may change.
    string name;          ///< Symbolic name, empty when there is none.
};
using DimDescs = vector<DimDesc>;

class TensorObj : public TensorBaseObj {
  private:
    Shape shape;
    size_t _size; // Cache of Π(shape).
    Fuid fuid;    // Cloned tensors share the same id. Tensors constructed from
                  // scratch have a new id.
    TensorType tensorType = TensorType::others;
    // Per-dimension dynamicity. Empty means the tensor never declared which
    // dimensions are dynamic, in which case its shape may be replaced freely.
    DimDescs dimDescs;
    // Contents of an integer tensor that describes shapes, worked out during
    // shape inference. See `getShapeValue`.
    optional<vector<int64_t>> shapeValue;
    // Which elements of `shapeValue` are the same under every shape the graph
    // may legally be given. Always as long as `shapeValue` when that holds a
    // value, because both are written together. See `isShapeValueFixed`.
    vector<bool> shapeValueFixed;

  public:
    TensorObj(Shape shape, DataType dtype, Runtime runtime);
    virtual ~TensorObj() {}
    string toString() const override;

    size_t size() const { return _size; }
    size_t getBytes() const { return _size * dtype.getSize(); }

    Shape getDims() const { return shape; }
    void setShape(Shape shape_);
    size_t getRank() const { return shape.size(); }

    /// @brief Declares which dimensions are dynamic. Once declared, the fixed
    /// dimensions are protected by `validateShapeChange`.
    void setDimDescs(DimDescs descs);
    const DimDescs &getDimDescs() const { return dimDescs; }
    /// @brief Whether dimension `dim` may change. Tensors without declared
    /// dimensionality report every dimension as dynamic, matching the
    /// historical behavior of replacing their shape freely.
    bool isDimDynamic(size_t dim) const;
    /// @brief Symbolic name of dimension `dim`, empty when it has none.
    string getDimName(size_t dim) const;
    /// @brief Throws when `target` violates the declared dimensionality.
    /// Does nothing for tensors that never declared it.
    void validateShapeChange(const Shape &target) const;

    /// @brief Contents of this tensor, when they describe shapes and are known
    /// without executing anything.
    ///
    /// Operators that only rearrange dimensions -- `Shape`, `Gather`,
    /// `Concat` and the like -- can be worked out during shape inference,
    /// because their result follows from the shapes of their inputs. Shape
    /// inference stores that result here, so that a `Reshape` reading its
    /// target shape from a tensor can find it. Empty when the contents are
    /// unknown, which is the case for ordinary data.
    ///
    /// The value describes the current shapes, so it is rewritten whenever
    /// shape inference runs again.
    const optional<vector<int64_t>> &getShapeValue() const {
        return shapeValue;
    }
    /// @brief Records `value` as following only from fixed dimensions.
    ///
    /// This is the right reading for the contents of a constant, which are
    /// whatever the model file says whatever shape the graph is given.
    void setShapeValue(vector<int64_t> value);
    /// @brief Records `value` along with which of its elements are the same
    /// under every shape the graph may legally be given.
    void setShapeValue(vector<int64_t> value, vector<bool> fixed);
    void clearShapeValue() {
        shapeValue.reset();
        shapeValueFixed.clear();
    }
    /// @brief Whether element `index` of the shape value is the same under
    /// every shape the graph may legally be given.
    ///
    /// A dimension the model declared fixed cannot change: `set_input` goes
    /// through `validateShapeChange`, which rejects any attempt to. So an
    /// element that follows only from such dimensions is already its final
    /// value while the graph is being built, and whatever computes it need not
    /// be run again -- or kept at all.
    bool isShapeValueFixed(size_t index) const;
    /// @brief Whether every element of the shape value is fixed.
    /// False when there is no shape value to speak of.
    bool isShapeValueWhollyFixed() const;
    /// @brief The shape value narrowed to `Shape`, checking every element fits.
    Shape getShapeValueAsShape() const;
    /// @brief Whether this tensor is shaped and typed like a dimension list.
    ///
    /// A shape is a list of integers, so anything of a wider rank or of a
    /// different type holds data rather than dimensions.
    bool canHoldShapeValue() const;

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
        obj->clearCaptureStates();
        obj->freeData();
        obj->targets.clear();
        obj->source.reset();
        return obj;
    }
    Tensor clone(Runtime runtime) const {
        auto obj = make_ref<TensorObj>(*this);
        obj->clearCaptureStates();
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
    //             data[i] = fastrand(random_seed[omp_get_thread_num() *
    //             16]) % 10000;
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
    //             if (ds[nBroadcastDim + i] < 0 || ds[nBroadcastDim +
    //             i] >= dims[i])
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

    //     std::vector<std::vector<int>> const *getSplittingPoints()
    //     const {
    //         assert(!splittingPoints.empty());
    //         return &splittingPoints;
    //     }

    //     bool setSplittingPoints(std::vector<std::vector<int>> value)
    //     {
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
    //     splittingPoints.resize(getRank()); }

    //     void printShape();
};

} // namespace infini
