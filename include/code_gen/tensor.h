#ifndef TENSOR_H
#define TENSOR_H

#include "common.h"
#include "dim.h"

namespace tpm {

class Tensor {
  public:
    enum DataType {
        Float32,
        Int32,
    };

    enum TensorType {
        Input,
        Weight,
        Invalid,
        NotCounted,
    };

    // TODO: is more compute state needed?
    enum ComputeState {
        NotComputed,
        // Allocated,
        // Initialized,
        // ComputedPartial,
        ComputedFull,
    };

  private:
    const size_t guid;
    uint64_t hash;
    Dim dims;
    OpVec inputOf;
    Operator *outputOf;
    VType *data;
    Dim it;
    DataType dtype;
    TensorType type;
    ComputeState computed;
    static int random_seed[256 * 16];
    static bool random_inited;

    // splitting points [dim][n-th splitting point]
    std::vector<std::vector<int>> splittingPoints;

    Dim dimPenalty;

  public:
    Tensor(TensorType type = Input, DataType dtype = Float32)
        : guid(generateGuid()), hash(generateHash()), outputOf(nullptr),
          data(nullptr), dtype(dtype), type(type), computed(NotComputed) {}
    Tensor(const Dim &dims, TensorType type = Input, DataType dtype = Float32)
        : guid(generateGuid()), hash(generateHash()), dims(dims),
          outputOf(nullptr), data(nullptr), dtype(dtype), type(type),
          computed(NotComputed) {
        itInit();
    }
    Tensor(const Tensor &rhs) : Tensor(rhs.dims, rhs.type, rhs.dtype) {
        outputOf = nullptr;
        data = nullptr;
        hash = rhs.hash;
        dimPenalty = rhs.dimPenalty;
        itInit();
    }
    Tensor(VType scalar, TensorType type = Weight, DataType dtype = Float32)
        : guid(generateGuid()), hash(generateHash()), outputOf(nullptr),
          data(nullptr), dtype(dtype), type(type), computed(ComputedFull) {
        assert(size() == 1);
        dataMalloc();
        data[0] = scalar;
    }
    ~Tensor() {
        if (data != nullptr)
            delete[] data;
    }

    // inputOf and outputOf will not be cloned
    Tensor *clone() {
        Tensor *t = new Tensor(*this);
        return t;
    }

    void clone(Tensor *t) {
        dims = t->dims;
        dtype = t->dtype;
        type = t->type;
        hash = t->hash;
        dimPenalty = t->dimPenalty;
    }

    DataType getDType() const { return dtype; }

    size_t getGuid() const { return guid; }

    void replace(Tensor &t) { hash = t.hash; }
    void refresh() { hash = generateHash(); }
    uint64_t getHash() const { return hash; }

    const Dim &getDims() const { return dims; }

    void setDims(const Dim &dms) { dims = dms; }

    void setInputOf(const OpVec &vec) { inputOf = vec; }

    void addInputOf(Operator *op) { inputOf.emplace_back(op); }
    void setOutputOf(Operator *op) { outputOf = op; }

    // TODO: more tensor state
    // if tensor is clear
    bool isClear() {
        return inputOf.empty() && outputOf == nullptr && type == Input &&
               computed == NotComputed && splittingPoints.empty();
    }
    // set tensor to clear state
    void clear() {
        inputOf.clear();
        outputOf = nullptr;
        type = Input;
        computed = NotComputed;
        splittingPoints.clear();
        hash = generateHash();
        dimPenalty.clear();
    }

    bool isComputed() const { return computed == ComputedFull; }
    void setComputed() { computed = ComputedFull; }

    bool isScalar() const { return dims.empty(); }

    bool isValid() const { return type != Invalid; }
    void setInvalid() { type = Invalid; }

    bool isNotCounted() const { return type == NotCounted; }

    void resetPenalty() {
        for (auto &i : dimPenalty)
            i = 0;
        dimPenalty.resize(dims.size(), 0);
    }
    const Dim &getPenalty() {
        if (dimPenalty.empty())
            dimPenalty.resize(dims.size(), 0);
        return dimPenalty;
    }
    Dim getPenalty() const {
        return dimPenalty.empty() ? Dim(dims.size(), 0) : dimPenalty;
    }
    void addPenalty(int d, int penalty = 1) {
        if (dimPenalty.empty())
            dimPenalty.resize(dims.size(), 0);
        dimPenalty[d] += penalty;
    }
    void setPenalty(const Dim &penalty) {
        dimPenalty.resize(penalty.size());
        dimPenalty = penalty;
    }

    const OpVec &getInputOf() { return inputOf; }
    Operator *getOutputOf() { return outputOf; }
    std::pair<Operator *, int> getOutputOfWithIndex();

    bool dataMalloc() {
        if (data == nullptr)
            data = new VType[size()];
        return data != nullptr;
    }

    bool dataRand(int seed = 0) {
        if (data == nullptr)
            data = new VType[size()];
        if (!random_inited)
            initFastrand();
        // srand(seed);
        // faster rand generator; parallel
        size_t iEnd = size();
        // std::cerr << "Init beginned " << std::endl;
#pragma omp parallel for
        for (size_t i = 0; i < iEnd; ++i)
            data[i] = fastrand(random_seed[omp_get_thread_num() * 16]) % 10000;
        // std::cerr << "Init finished" << std::endl;
        computed = ComputedFull;
        return true;
    }

    bool setData(VType *dptr) {
        if (dptr == nullptr)
            return false;
        auto sz = size();
#pragma omp parallel for
        for (size_t i = 0; i < sz; ++i)
            data[i] = dptr[i];
        computed = ComputedFull;
        return true;
    }

    bool setScalar(VType val) {
        if (data == nullptr || !dims.empty())
            return false;
        data[0] = val;
        return true;
    }

    bool setData(const Dim &ds, VType val) {
        if (data == nullptr || ds.size() != dims.size())
            return false;
        data[getOffset(ds)] = val;
        return true;
    }

    bool setData(size_t pos, VType val) {
        if (data == nullptr || pos >= size())
            return false;
        data[pos] = val;
        return true;
    }

    VType getScalar() { return data == nullptr ? 0 : data[0]; }

    VType getData(const Dim &ds) {
        assert(data != nullptr);
        auto offset = getOffset(ds);
        return offset == (size_t)-1 ? 0 : data[getOffset(ds)];
    }

    VType getData(size_t pos) {
        assert(data != nullptr);
        assert(pos < size());
        return data[pos];
    }

    VType *getDataPtr() const { return data; }

    size_t getOffset(const Dim &ds) {
        auto nDim = ds.size();
        assert(dims.size() == nDim);
        if (ds.empty())
            return 0;
        for (size_t i = 0; i < nDim; ++i)
            if (ds[i] < 0 || ds[i] >= dims[i])
                return (size_t)-1;
        size_t idx = ds[0];
        size_t dm = 0;
        while (++dm < nDim)
            idx = idx * dims[dm] + ds[dm];
        return idx;
    }

    VType getBroadcastData(const Dim &ds) {
        assert(data != nullptr);
        auto offset = getBroadcastOffset(ds);
        return offset == (size_t)-1 ? 0 : data[getOffset(ds)];
    }

    VType getBroadcastData(size_t pos) {
        assert(data != nullptr);
        return data[pos % size()];
    }

    size_t getBroadcastOffset(const Dim &ds) {
        assert(ds.size() >= dims.size());
        auto nDim = dims.size();
        auto nBroadcastDim = ds.size() - nDim;
        for (size_t i = 0; i < nDim; ++i)
            if (ds[nBroadcastDim + i] < 0 || ds[nBroadcastDim + i] >= dims[i])
                return (size_t)-1;
        size_t idx = 0;
        for (size_t i = 0; i < nDim; ++i)
            idx = idx * dims[i] + ds[nBroadcastDim + i];
        return idx;
    }

    void itInit() { it = Dim(dims.size(), 0); }

    void itReset() {
        itInit();
        for (size_t i = 0, iEnd = it.size(); i < iEnd; ++i)
            it[i] = 0;
    }

    bool itValid() {
        if (it.size() != dims.size())
            return false;
        for (size_t i = 0, iEnd = it.size(); i < iEnd; ++i)
            if (it[i] >= dims[i])
                return false;
        return true;
    }

    const Dim &itGet() { return it; }

    void itNext() {
        auto p = it.size() - 1;
        it[p] += 1;
        while (p >= 1) {
            if (it[p] == dims[p]) {
                it[p] = 0;
                it[--p] += 1;
            } else
                break;
        }
    }

    size_t size() const {
        size_t sz = 1;
        auto dm = dims.size();
        while (dm > 0)
            sz *= dims[--dm];
        return sz;
    }

    TensorType getType() const { return type; }
    void setType(TensorType ty) { type = ty; }

    void print() {
        if (type == Invalid) {
            std::cout << "Invalid tensor" << std::endl;
            return;
        }

        if (data == nullptr || dims.size() == 0) {
            std::cout << "Empty tensor" << std::endl;
            return;
        }

        // TODO: can be uncommented after tensor's compute type is correctly set
        if (computed == NotComputed) {
            std::cout << "Uncomputed tensor" << std::endl;
            return;
        }

        std::cout << "Tensor: " << guid << std::endl;
        auto numDims = dims.size();
        auto dimSzVec = std::vector<int>(numDims, 1);
        dimSzVec[numDims - 1] = dims[numDims - 1];
        for (int i = numDims - 1; i != 0; --i)
            dimSzVec[i - 1] = dimSzVec[i] * dims[i - 1];
        for (size_t i = 0, iEnd = size(); i < iEnd; ++i) {
            for (size_t j = 0; j < numDims; ++j) {
                if (i % dimSzVec[j] == 0) {
                    std::cout << "[";
                }
            }
            std::cout << data[i];
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

    static inline void initFastrand() {
        assert(omp_get_max_threads() <= 256);
        // srand(0); // constant seed for test
        // align random_seed to avoid false sharing
        for (int i = 0; i < 256 * 16; ++i) {
            // random_seed[i] = rand();
            // constant random seed for test
            random_seed[i] = i;
        }
        random_inited = true;
    }

    static inline int fastrand(int &g_seed) {
        g_seed = (214013 * g_seed + 2531011);
        return (g_seed >> 16) & 0x7FFF;
    }

    std::vector<std::vector<int>> const *getSplittingPoints() const {
        assert(!splittingPoints.empty());
        return &splittingPoints;
    }

    bool setSplittingPoints(std::vector<std::vector<int>> value) {
        assert(!value.empty());
        splittingPoints = value;
        return true;
    }

    void printSplittingPoints() {
        if (splittingPoints.empty())
            printf("Empty SplittingPoints");
        else {
            printf("[");
            for (auto &vs : splittingPoints) {
                printf("[");
                for (auto v : vs)
                    printf("%2d,", v);
                printf("],");
            }
            printf("]");
        }
    }

    void initSplittingPoints() { splittingPoints.resize(getDims().size()); }

    void printShape();
};

void printTensor(tpm::Tensor *tensor);

} // end of namespace tpm

#endif // TENSOR_H
