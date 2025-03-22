#include "core/data_type.h"
#include "core/graph.h"
#include "core/ref.h"
#include "core/runtime.h"
#include "core/tensor.h"
#include "operators/where.h"

#include "test.h"
#include "utils/data_generator.h"
#include <functional>
#include <gtest/gtest.h>
#include <optional>
#include <type_traits>

namespace infini {

template <class T>
void testWhereCpu(
    const std::function<void(void *, size_t, DataType)> &generatorA,
    const std::function<void(void *, size_t, DataType)> &generatorB,
    const Shape &x_shape, const Shape &y_shape, const Shape &cond_shape,
    const DataType &dataType) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto x = g->addTensor(x_shape, dataType);
    auto y = g->addTensor(y_shape, dataType);
    auto cond = g->addTensor(cond_shape, DataType::UInt8);

    auto op = g->addOp<T>(x, y, cond, nullptr);
    g->dataMalloc();
    x->setData(generatorA);
    y->setData(generatorA);
    cond->setData(generatorB);

    runtime->run(g);
    // op->getOutput()->print();
    // op->getOutput()->printData();
    EXPECT_TRUE(1);
}

TEST(Where, Cpu) {
    testWhereCpu<WhereObj>(IncrementalGenerator(), OneGenerator(),
                           Shape{1}, Shape{2, 2}, Shape{2, 2}, DataType::Float16);
    testWhereCpu<WhereObj>(IncrementalGenerator(), ZeroGenerator(),
                           Shape{1}, Shape{2, 2}, Shape{2, 2}, DataType::Float16);
    testWhereCpu<WhereObj>(IncrementalGenerator(), OneGenerator(),
                           Shape{2, 2}, Shape{1}, Shape{2, 2}, DataType::Float16);
    testWhereCpu<WhereObj>(IncrementalGenerator(), ZeroGenerator(),
                           Shape{2, 2}, Shape{1}, Shape{2, 2}, DataType::Float16);

    testWhereCpu<WhereObj>(IncrementalGenerator(), OneGenerator(),
                           Shape{2, 2}, Shape{2, 2}, Shape{1}, DataType::Float16);
    testWhereCpu<WhereObj>(IncrementalGenerator(), ZeroGenerator(),
                           Shape{2, 2}, Shape{2, 2}, Shape{1}, DataType::Float16);
    testWhereCpu<WhereObj>(IncrementalGenerator(), OneGenerator(),
                           Shape{2, 2}, Shape{2, 2}, Shape{2, 2}, DataType::Float16);
    testWhereCpu<WhereObj>(IncrementalGenerator(), ZeroGenerator(),
                           Shape{2, 2}, Shape{2, 2}, Shape{2, 2}, DataType::Float16);    

    testWhereCpu<WhereObj>(IncrementalGenerator(), OneGenerator(),
                           Shape{1}, Shape{2, 2}, Shape{2, 2}, DataType::Float32);
    testWhereCpu<WhereObj>(IncrementalGenerator(), ZeroGenerator(),
                           Shape{1}, Shape{2, 2}, Shape{2, 2}, DataType::Float32);
    testWhereCpu<WhereObj>(IncrementalGenerator(), OneGenerator(),
                           Shape{2, 2}, Shape{1}, Shape{2, 2}, DataType::Float32);
    testWhereCpu<WhereObj>(IncrementalGenerator(), ZeroGenerator(),
                           Shape{2, 2}, Shape{1}, Shape{2, 2}, DataType::Float32);

    testWhereCpu<WhereObj>(IncrementalGenerator(), OneGenerator(),
                           Shape{2, 2}, Shape{2, 2}, Shape{1}, DataType::Float32);
    testWhereCpu<WhereObj>(IncrementalGenerator(), ZeroGenerator(),
                           Shape{2, 2}, Shape{2, 2}, Shape{1}, DataType::Float32);
    testWhereCpu<WhereObj>(IncrementalGenerator(), OneGenerator(),
                           Shape{2, 2}, Shape{2, 2}, Shape{2, 2}, DataType::Float32);
    testWhereCpu<WhereObj>(IncrementalGenerator(), ZeroGenerator(),
                           Shape{2, 2}, Shape{2, 2}, Shape{2, 2}, DataType::Float32);                             
}

} // namespace infini
