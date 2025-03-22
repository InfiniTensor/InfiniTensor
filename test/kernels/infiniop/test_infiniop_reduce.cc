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

template <class T>
void testReduceCpu(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &x_shape, const DataType &dataType, std::vector<int> axes,
    bool keepdims) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto x = g->addTensor(x_shape, dataType);
    auto op = g->addOp<T>(x, nullptr, axes, keepdims);

    g->dataMalloc();
    x->setData(generator);

    runtime->run(g);
    // op->getOutput()->print();
    // op->getOutput()->printData();
    EXPECT_TRUE(1);
}

TEST(Reduce, Cpu) {
    testReduceCpu<ReduceMinObj>(IncrementalGenerator(), Shape{3, 2}, DataType::Float16,
                               {0}, true);
    testReduceCpu<ReduceMinObj>(IncrementalGenerator(), Shape{3, 2}, DataType::Float32,
                               {0}, true);
    testReduceCpu<ReduceMaxObj>(IncrementalGenerator(), Shape{3, 2, 3}, DataType::Float16,
                               {0}, true);
    testReduceCpu<ReduceMaxObj>(IncrementalGenerator(), Shape{3, 2, 3}, DataType::Float32,
                               {0}, true);
    testReduceCpu<ReduceMeanObj>(IncrementalGenerator(), Shape{3, 2, 3}, DataType::Float16,
                               {0}, true);
    testReduceCpu<ReduceMeanObj>(IncrementalGenerator(), Shape{3, 2, 3}, DataType::Float32,
                               {0}, true);                               

    testReduceCpu<ReduceMinObj>(IncrementalGenerator(), Shape{3, 2}, DataType::Float16,
                               {0}, false);
    testReduceCpu<ReduceMinObj>(IncrementalGenerator(), Shape{3, 2}, DataType::Float32,
                               {0}, false);
    testReduceCpu<ReduceMaxObj>(IncrementalGenerator(), Shape{3, 2, 3}, DataType::Float16,
                               {0}, false);
    testReduceCpu<ReduceMaxObj>(IncrementalGenerator(), Shape{3, 2, 3}, DataType::Float32,
                               {0}, false);
    testReduceCpu<ReduceMeanObj>(IncrementalGenerator(), Shape{3, 2, 3}, DataType::Float16,
                               {0}, false);
    testReduceCpu<ReduceMeanObj>(IncrementalGenerator(), Shape{3, 2, 3}, DataType::Float32,
                               {0}, false);                                    
}

} // namespace infini
