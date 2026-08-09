#pragma once
#include "core/common.h"
#include "core/tensor_base.h"
#include "utils/data_convert.h"
#include <random>

namespace infini {

// Lightweight test data generators used by unit tests and examples.
// Generators support multiple DataType variants; Bool is stored as
// `int8_t` in this project so generators expose an `int8_t` overload.
// Keep implementations simple and deterministic for tests.
// TODO: isolate these class
class DataGenerator {
  private:
    virtual void fill(uint32_t *data, size_t size) { IT_TODO_HALT(); }
    virtual void fill(float *data, size_t size) { IT_TODO_HALT(); }
    virtual void fill_fp16(uint16_t *data, size_t size) { IT_TODO_HALT(); }
    // DataType::Bool is stored as int8_t (1 byte) in this project (see
    // include/core/data_type.h). Provide a fill overload for that.
    virtual void fill(int8_t *data, size_t size) { IT_TODO_HALT(); }

  public:
    virtual ~DataGenerator() {}
    void operator()(void *data, size_t size, DataType dataType) {
        if (dataType == DataType::UInt32)
            fill(reinterpret_cast<uint32_t *>(data), size);
        else if (dataType == DataType::Float32)
            fill(reinterpret_cast<float *>(data), size);
        else if (dataType == DataType::Float16)
            fill_fp16(reinterpret_cast<uint16_t *>(data), size);
        else if (dataType == DataType::Bool)
            // Bool is stored as 1 byte (int8_t) according to DataType::getSize
            // / sizePerElement. Use the int8_t overload.
            fill(reinterpret_cast<int8_t *>(data), size);
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
    void fill_fp16(uint16_t *data, size_t size) override {
        for (size_t i = 0; i < size; i++) {
            float x = 2.0f;
            data[i] = float_to_fp16(x);
        }
    }
    void fill(int8_t *data, size_t size) override {
        for (size_t i = 0; i < size; i++)
            data[i] = static_cast<int8_t>(i);
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
    void fill(int8_t *data, size_t size) override {
        for (size_t i = 0; i < size; i++)
            data[i] = static_cast<int8_t>(di(e));
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
    void fill_fp16(uint16_t *data, size_t size) override {
        for (size_t i = 0; i < size; i++) {
            float x = 1.0f * val;
            data[i] = float_to_fp16(x);
        }
    }
    void fill(int8_t *data, size_t size) override {
        for (size_t i = 0; i < size; i++)
            data[i] = static_cast<int8_t>(val);
    }
};
typedef ValGenerator<1> OneGenerator;
typedef ValGenerator<0> ZeroGenerator;
} // namespace infini
