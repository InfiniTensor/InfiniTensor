#pragma once
#include "core/common.h"
#include "core/tensor_base.h"
#include <random>

namespace infini {

// TODO: isolate these class
class DataGenerator {
  private:
    virtual void fill(uint32_t *data, size_t size) { IT_TODO_HALT(); }
    virtual void fill(float *data, size_t size) { IT_TODO_HALT(); }

  public:
    virtual ~DataGenerator() {}
    void operator()(void *data, size_t size, DataType dataType) {
        if (dataType == DataType::UInt32)
            fill(reinterpret_cast<uint32_t *>(data), size);
        else if (dataType == DataType::Float32)
            fill(reinterpret_cast<float *>(data), size);
        else
            IT_TODO_HALT();
    }
};

class IncrementalGenerator : public DataGenerator {
  public:
    virtual ~IncrementalGenerator() {}

  private:
    template <typename T> void fill(T *data, size_t size) {
        for (size_t i = 0; i < size; i++) {
            data[i] = i;
        }
    }

    void fill(uint32_t *data, size_t size) override {
        fill<uint32_t>(data, size);
    }
    void fill(float *data, size_t size) override { fill<float>(data, size); }
};

class RandomGenerator : public DataGenerator {
  private:
    double l, r;
    std::mt19937 e;
    std::uniform_int_distribution<int> di;
    std::uniform_real_distribution<float> dr;
    bool generateInteger;

  public:
    RandomGenerator(double l = 0, double r = 1, unsigned int seed = 0,
                    bool generateInteger = false)
        : l(l), r(r), e(seed), di(l, r), dr(l, r),
          generateInteger(generateInteger) {}
    virtual ~RandomGenerator() {}

  private:
    void fill(uint32_t *data, size_t size) override {
        for (size_t i = 0; i < size; i++) {
            data[i] = di(e);
        }
    }
    void fill(float *data, size_t size) override {
        for (size_t i = 0; i < size; i++) {
            data[i] = (generateInteger) ? di(e) : dr(e);
        }
    }
};

template <int val> class ValGenerator : public DataGenerator {
  public:
    virtual ~ValGenerator() {}

  private:
    template <typename T> void fill(T *data, size_t size) {
        for (size_t i = 0; i < size; i++) {
            data[i] = val;
        }
    }

    void fill(uint32_t *data, size_t size) override {
        fill<uint32_t>(data, size);
    }
    void fill(float *data, size_t size) override { fill<float>(data, size); }
};
typedef ValGenerator<1> OneGenerator;
typedef ValGenerator<0> ZeroGenerator;
} // namespace infini
