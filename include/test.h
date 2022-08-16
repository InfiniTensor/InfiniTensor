#pragma once
#include "core/common.h"
#include "gtest/gtest.h"

namespace infini {

class DataGenerator {
  private:
    virtual void fill(int *data, size_t size) { IT_TODO_HALT(); };
    virtual void fill(float *data, size_t size) { IT_TODO_HALT(); };

  public:
    void operator()(void *data, size_t size, DataType dataType) {
        switch (dataType) {
        case DataType::Int32:
            fill(reinterpret_cast<int *>(data), size);
            break;
        case DataType::Float32:
            fill(reinterpret_cast<float *>(data), size);
            break;
        default:
            IT_TODO_HALT();
        }
    }
};

class IncrementalGenerator : public DataGenerator {
    void fill(int *data, size_t size) override {
        for (size_t i = 0; i < size; i++) {
            data[i] = i;
        }
    }
};
} // namespace infini