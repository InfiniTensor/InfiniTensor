#include "core/data_type.h"
#include "core/graph.h"
#include "core/ref.h"
#include "core/runtime.h"
#include "core/tensor.h"
#include "operators/reduce.h"

#include "test.h"
#include "utils/data_generator.h"
#include <functional>
#include <gtest/gtest.h>
#include <optional>
#include <type_traits>

namespace infini {

template <class T, typename std::enable_if<std::is_base_of<ReduceBaseObj, T>{},
                                           int>::type = 0>
void testReduceCpu(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &x_shape, const DataType &dataType,
    std::vector<int> axes, bool keepDims) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto x = g->addTensor(x_shape, dataType);

    auto op = g->addOp<T>(x, nullptr, axes, keepDims);
    g->dataMalloc();
    x->setData(generator);
    // x->print();
    // x->printData();

    runtime->run(g);
    // op->getOutput()->print();
    // op->getOutput()->printData();
    EXPECT_TRUE(1);
}

TEST(Reduce, Cpu) {
    testReduceCpu<ReduceMinObj>(IncrementalGenerator(), Shape{3, 4},
                                DataType::Float16, {0}, true);
    testReduceCpu<ReduceMinObj>(IncrementalGenerator(), Shape{3, 4},
                                DataType::Float32, {0}, true);
    testReduceCpu<ReduceMinObj>(IncrementalGenerator(), Shape{3, 4, 5},
                                DataType::Float16, {1, 2}, false);
    testReduceCpu<ReduceMinObj>(IncrementalGenerator(), Shape{3, 4, 5},
                                DataType::Float32, {1, 2}, false);
    testReduceCpu<ReduceMaxObj>(IncrementalGenerator(), Shape{3, 4},
                                DataType::Float16, {0}, true);
    testReduceCpu<ReduceMaxObj>(IncrementalGenerator(), Shape{3, 4},
                                DataType::Float32, {0}, true);
    testReduceCpu<ReduceMaxObj>(IncrementalGenerator(), Shape{3, 4, 5},
                                DataType::Float16, {1, 2}, false);
    testReduceCpu<ReduceMaxObj>(IncrementalGenerator(), Shape{3, 4, 5},
                                DataType::Float32, {1, 2}, false);
    testReduceCpu<ReduceMeanObj>(IncrementalGenerator(), Shape{3, 4},
                                 DataType::Float16, {0}, true);
    testReduceCpu<ReduceMeanObj>(IncrementalGenerator(), Shape{3, 4},
                                 DataType::Float32, {0}, true);
    testReduceCpu<ReduceMeanObj>(IncrementalGenerator(), Shape{3, 4, 5},
                                 DataType::Float16, {1, 2}, false);
    testReduceCpu<ReduceMeanObj>(IncrementalGenerator(), Shape{3, 4, 5},
                                 DataType::Float32, {1, 2}, false);
}

}
