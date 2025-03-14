#pragma once
#include "core/common.h"
#include "core/tensor_base.h"
#include "utils/data_convert.h"
#include <random>

namespace infini {

// TODO: isolate these class
class DataGenerator {
  private:
    virtual void fill(uint32_t *data, size_t size) { IT_TODO_HALT(); }
    virtual void fill(float *data, size_t size) { IT_TODO_HALT(); }
    virtual void fill(int32_t *data, size_t size) { IT_TODO_HALT(); };
    virtual void fill(bool *data, size_t size) { IT_TODO_HALT(); };
    virtual void fill(uint8_t *data, size_t size) { IT_TODO_HALT(); };
    virtual void fill(int64_t *data, size_t size) { IT_TODO_HALT(); };
    virtual void fill_fp16(uint16_t *data, size_t size) { IT_TODO_HALT(); }

  public:
    virtual ~DataGenerator() {}
    void operator()(void *data, size_t size, DataType dataType) {
        if (dataType == DataType::UInt32)
            fill(reinterpret_cast<uint32_t *>(data), size);
        else if(dataType == DataType::Int64)
            fill(reinterpret_cast<int64_t *>(data), size);
        else if(dataType == DataType::UInt8)
            fill(reinterpret_cast<uint8_t *>(data), size);
        else if(dataType == DataType::Bool)
            fill(reinterpret_cast<bool *>(data), size);
        else if (dataType == DataType::Float32)
            fill(reinterpret_cast<float *>(data), size);
        else if(dataType == DataType::Int32)
            fill(reinterpret_cast<int32_t *>(data), size);
        else if (dataType == DataType::Float16)
            fill_fp16(reinterpret_cast<uint16_t *>(data), size);
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
    // FIXME: fix the accuracy standards when dtype is float16
    void fill(int32_t *data, size_t size) override {
        fill<int32_t>(data, size);
    }
    void fill(uint8_t *data, size_t size) override {
        fill<uint8_t>(data, size);
    }
    void fill(int64_t *data, size_t size) override {
        fill<int64_t>(data, size);
    }
    void fill_fp16(uint16_t *data, size_t size) {
        for (size_t i = 0; i < size; i++) {
            float x = 2.0f;
            data[i] = float_to_fp16(x);
        }
    }
    void fill(bool *data, size_t size) override {
        fill<bool>(data, size);
    }
};

class RandomGenerator : public DataGenerator {
  private:
    double l, r;
    std::mt19937 e;
    std::uniform_int_distribution<int> di;
    std::uniform_real_distribution<float> dr;

  public:
    RandomGenerator(double l = 0, double r = 1, unsigned int seed = 0)
        : l(l), r(r), e(seed), di(l, r), dr(l, r) {}
    virtual ~RandomGenerator() {}

  private:
    void fill(uint32_t *data, size_t size) override {
        for (size_t i = 0; i < size; i++) {
            data[i] = di(e);
        }
    }
    void fill(float *data, size_t size) override {
        for (size_t i = 0; i < size; i++) {
            data[i] = dr(e);
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
    void fill_fp16(uint16_t *data, size_t size) {
        for (size_t i = 0; i < size; i++) {
            float x = 1.0f * val;
            data[i] = float_to_fp16(x);
        }
    }
};
typedef ValGenerator<1> OneGenerator;
typedef ValGenerator<0> ZeroGenerator;
} // namespace infini
