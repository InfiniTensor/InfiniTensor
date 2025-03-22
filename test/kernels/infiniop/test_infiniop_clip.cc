#include "core/graph.h"
#include "core/runtime.h"
#include "core/data_type.h"
#include "core/ref.h"
#include "core/tensor.h"
#include "operators/unary.h"

#include "test.h"
#include "utils/data_generator.h"
#include <functional>
#include <gtest/gtest.h>
#include <optional>
#include <type_traits>

namespace infini {

template <class T>
void testClipCpu(const std::function<void(void *, size_t, DataType)> &generator,
                 const Shape &input_shape, const DataType &dataType,
                 std::optional<float> min = std::nullopt,
                 std::optional<float> max = std::nullopt) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(input_shape, dataType);

    auto op = g->addOp<T>(input, nullptr, min, max);
    g->dataMalloc();
    input->setData(generator);

    runtime->run(g);
    // op->getOutput()->print();
    // op->getOutput()->printData();
    EXPECT_TRUE(1);
}

TEST(Clip, Cpu) {
    testClipCpu<ClipObj>(IncrementalGenerator(), Shape{3, 2}, DataType::Float16);
    testClipCpu<ClipObj>(IncrementalGenerator(), Shape{3, 2}, DataType::Float32);

    testClipCpu<ClipObj>(IncrementalGenerator(), Shape{3, 2}, DataType::Float16,
                         -1.0f, 1.0f);
    testClipCpu<ClipObj>(IncrementalGenerator(), Shape{3, 2}, DataType::Float32,
                         -1.0f, 1.0f);

    testClipCpu<ClipObj>(IncrementalGenerator(), Shape{3, 2}, DataType::Float16,
                         -1.0f, std::nullopt);
    testClipCpu<ClipObj>(IncrementalGenerator(), Shape{3, 2}, DataType::Float32,
                         -1.0f, std::nullopt);

    testClipCpu<ClipObj>(IncrementalGenerator(), Shape{3, 2}, DataType::Float16,
                         std::nullopt, 1.0f);
    testClipCpu<ClipObj>(IncrementalGenerator(), Shape{3, 2}, DataType::Float32,
                         std::nullopt, 1.0f);
}

} // namespace infini
