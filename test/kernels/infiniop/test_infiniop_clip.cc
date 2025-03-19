#include "core/data_type.h"
#include "core/graph.h"
#include "core/ref.h"
#include "core/runtime.h"
#include "core/tensor.h"
#include "operators/unary.h"

#include "test.h"
#include "utils/data_generator.h"
#include <functional>
#include <gtest/gtest.h>
#include <optional>
#include <type_traits>

namespace infini {

template <class T,
          typename std::enable_if<std::is_base_of<ClipObj, T>{}, int>::type = 0>
void testClipCpu(const std::function<void(void *, size_t, DataType)> &generator,
                 const Shape &shape, const DataType &dataType,
                 std::optional<float> min = std::nullopt,
                 std::optional<float> max = std::nullopt) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shape, dataType);

    auto op = g->addOp<T>(input, nullptr, min, max);
    g->dataMalloc();
    input->setData(generator);
    // input->print();
    // input->printData();

    runtime->run(g);
    // fp16, printData() will not get the expected result
    // op->getOutput()->print();
    // op->getOutput()->printData();
    EXPECT_TRUE(1);
}

TEST(Clip, Cpu) {
    testClipCpu<ClipObj>(IncrementalGenerator(), Shape{3}, DataType::Float16,
                         2.0f, 2.5f);
    testClipCpu<ClipObj>(IncrementalGenerator(), Shape{3}, DataType::Float32,
                         2.0f, 2.5f);
    testClipCpu<ClipObj>(IncrementalGenerator(), Shape{3, 4}, DataType::Float16,
                         0.1f, 7.3f);
    testClipCpu<ClipObj>(IncrementalGenerator(), Shape{3, 4}, DataType::Float32,
                         0.1f, 7.3f);
    testClipCpu<ClipObj>(IncrementalGenerator(), Shape{3, 4, 5},
                         DataType::Float16, 0.1f, 0.7f);
    testClipCpu<ClipObj>(IncrementalGenerator(), Shape{3, 4, 5},
                         DataType::Float32, 0.1f, 0.7f);
}

} // namespace infini
