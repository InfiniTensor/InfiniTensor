#pragma once
#include "core/common.h"
#include "core/tensor_base.h"
#include "gtest/gtest.h"

namespace infini {

class DataGenerator {
  private:
    virtual void fill(uint32_t *data, size_t size) { IT_TODO_HALT(); };
    virtual void fill(float *data, size_t size) { IT_TODO_HALT(); };

  public:
    void operator()(void *data, size_t size, DataType dataType) {
        switch (dataType) {
        case DataType::UInt32:
            fill(reinterpret_cast<uint32_t *>(data), size);
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
    void fill(uint32_t *data, size_t size) override {
        for (size_t i = 0; i < size; i++) {
            data[i] = i;
        }
    }
};
} // namespace infini