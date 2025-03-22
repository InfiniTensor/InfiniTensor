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

template <class T>
void testGatherCpu(const std::function<void(void *, size_t, DataType)> &generatorA,
                   const std::function<void(void *, size_t, DataType)> &generatorB,
                   const Shape &input_shape, 
                   const Shape &indices_shape,
                   const DataType &dataType,
                   const DataType &indicesType, 
                   const int axis) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(input_shape, dataType);
    auto indices = g->addTensor(indices_shape, indicesType);

    auto op = g->addOp<T>(input, indices, nullptr, axis);
    g->dataMalloc();
    input->setData(generatorA);
    indices->setData(generatorB);

    runtime->run(g);
    // op->getOutput()->print();
    // op->getOutput()->printData();
    EXPECT_TRUE(1);
}

TEST(Gather, Cpu) {
    testGatherCpu<GatherObj>(IncrementalGenerator(), ZeroGenerator(), Shape{2, 3}, Shape{2},
                             DataType::Float16, DataType::Int32, 0);
    testGatherCpu<GatherObj>(IncrementalGenerator(), ZeroGenerator(), Shape{2, 3}, Shape{2},
                             DataType::Float32, DataType::Int32, 0);                         

    testGatherCpu<GatherObj>(IncrementalGenerator(), ZeroGenerator(), Shape{3, 4, 5}, Shape{2, 2},
                             DataType::Float16, DataType::Int32, 1);
    testGatherCpu<GatherObj>(IncrementalGenerator(), ZeroGenerator(), Shape{3, 4, 5}, Shape{2, 2},
                             DataType::Float32, DataType::Int32, 1);  
}

} // namespace infini
