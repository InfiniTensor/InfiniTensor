#pragma once
#include "core/common.h"
#include "core/tensor_base.h"
#include "gtest/gtest.h"

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
            data[i] = i+1;
        }
    }

    void fill(uint32_t *data, size_t size) override {
        fill<uint32_t>(data, size);
    }
    void fill(float *data, size_t size) override { fill<float>(data, size); }
};

class OneGenerator : public DataGenerator {
  public:
    virtual ~OneGenerator() {}

  private:
    template <typename T> void fill(T *data, size_t size) {
        for (size_t i = 0; i < size; i++) {
            data[i] = 1;
        }
    }

    void fill(uint32_t *data, size_t size) override {
        fill<uint32_t>(data, size);
    }
    void fill(float *data, size_t size) override { fill<float>(data, size); }
};
} // namespace infini
