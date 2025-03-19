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

template <class T, typename std::enable_if<std::is_base_of<WhereObj, T>{},
                                           int>::type = 0>
void testWhereCpu(
    const std::function<void(void *, size_t, DataType)> &generator1,
    const std::function<void(void *, size_t, DataType)> &generator2,
    const Shape &x_shape, const Shape &y_shape, const Shape &condition_shape,
    const DataType &x_dataType, const DataType &y_dataType) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto x = g->addTensor(x_shape, x_dataType);
    auto y = g->addTensor(y_shape, y_dataType);
    auto condition = g->addTensor(condition_shape, DataType::UInt8);

    auto op = g->addOp<T>(x, y, condition, nullptr);
    g->dataMalloc();
    x->setData(generator1);
    y->setData(generator1);
    condition->setData(generator2);

    // x->print();
    // x->printData();
    // y->print();
    // y->printData();
    // condition->print();
    // condition->printData();

    runtime->run(g);
    // op->getOutput()->print();
    // op->getOutput()->printData();
    EXPECT_TRUE(1);
}

TEST(Where, Cpu) {
    testWhereCpu<WhereObj>(IncrementalGenerator(), ZeroGenerator(), Shape{3,
    4},
                           Shape{3, 4}, Shape{3, 1}, DataType::Float16,
                           DataType::Float16);
    testWhereCpu<WhereObj>(IncrementalGenerator(), ZeroGenerator(), Shape{3,
    4},
                           Shape{3, 4}, Shape{3, 1}, DataType::Float32,
                           DataType::Float32);
    testWhereCpu<WhereObj>(IncrementalGenerator(), ZeroGenerator(), Shape{3,
    4, 5},
                           Shape{3, 4, 5}, Shape{3, 4, 1}, DataType::Float16,
                           DataType::Float16);
    testWhereCpu<WhereObj>(IncrementalGenerator(), ZeroGenerator(),
                           Shape{3, 4, 5}, Shape{3, 4, 5}, Shape{3, 4, 1},
                           DataType::Float32, DataType::Float32);
    testWhereCpu<WhereObj>(IncrementalGenerator(), OneGenerator(), Shape{3,
    4},
                           Shape{3, 1}, Shape{4}, DataType::Float16,
                           DataType::Float16);
    testWhereCpu<WhereObj>(IncrementalGenerator(), OneGenerator(), Shape{3,
    4},
                           Shape{3, 1}, Shape{4}, DataType::Float32,
                           DataType::Float32);
}

}
