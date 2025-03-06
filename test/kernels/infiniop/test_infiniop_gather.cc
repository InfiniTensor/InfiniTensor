#include "core/data_type.h"
#include "core/graph.h"
#include "core/ref.h"
#include "core/runtime.h"
#include "core/tensor.h"
#include "operators/gather.h"

#include "test.h"
#include "utils/data_generator.h"
#include <functional>
#include <gtest/gtest.h>
#include <optional>
#include <type_traits>

namespace infini {

template <class T, typename std::enable_if<std::is_base_of<GatherObj, T>{},
                                           int>::type = 0>
void testGatherCpu(
    const std::function<void(void *, size_t, DataType)> &generator1,
    const std::function<void(void *, size_t, DataType)> &generator2,
    const Shape &input_shape, const Shape &indices_shape,
    const DataType &input_dataType, const DataType &indices_dataType,
    const int axis) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(input_shape, input_dataType);
    auto indices = g->addTensor(indices_shape, indices_dataType);

    auto op = g->addOp<T>(input, indices, nullptr, axis);
    g->dataMalloc();
    input->setData(generator1);
    // int32 is not supported by IncrementalGenerator, so here use the zero indices
    indices->setData(generator2);

    // input->print();
    // input->printData();
    // indices->print();
    // indices->printData();

    runtime->run(g);
    // op->getOutput()->print();
    // op->getOutput()->printData();
    EXPECT_TRUE(1);
}

TEST(Gather, Cpu) {
    testGatherCpu<GatherObj>(IncrementalGenerator(), ZeroGenerator(),
                             Shape{3, 4}, Shape{2},
                             DataType::Float16, DataType::Int32, 0);
    testGatherCpu<GatherObj>(IncrementalGenerator(), ZeroGenerator(),
                             Shape{3, 4}, Shape{2}, DataType::Float32,
                             DataType::Int32, 0);
    testGatherCpu<GatherObj>(IncrementalGenerator(), ZeroGenerator(),
                             Shape{3, 4, 5}, Shape{2, 3}, DataType::Float16,
                             DataType::Int32, 1);
    testGatherCpu<GatherObj>(IncrementalGenerator(), ZeroGenerator(),
                             Shape{3, 4, 5}, Shape{2, 3}, DataType::Float32,
                             DataType::Int32, 1);
}

}
