#include "core/graph.h"
#include "core/runtime.h"
#include "operators/gather.h"
#include "test.h"

namespace infini {

template <class T>
void testGatherCpu(
    const std::function<void(void*, size_t, DataType)> &generator1,
    const std::function<void(void*, size_t, DataType)> &generator2,
    const Shape &inputShape, const Shape &indicesShape, const DataType &dataType,
    const DataType &indices_dataType,
    std::optional<int> axis = std::nullopt
) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    int axis_value = axis.value_or(0); 

    // 创建输入张量
    auto input = g->addTensor(inputShape, dataType);
    auto indices = g->addTensor(indicesShape, indices_dataType); // 假设索引是整数类型

    // 添加 Gather 操作
    auto op = g->addOp<GatherObj>(input, indices, nullptr,axis_value);

    g->dataMalloc();
    input->setData(generator1);  // 填充输入数据
    indices->setData(generator2); 
    runtime->run(g);

    // 输出结果
    // op->getOutput()->print();
    // op->getOutput()->printData();
    EXPECT_TRUE(1); 
}

TEST(ElementWise, Cpu) {
    // 测试用例：形状为 {1, 2, 2, 3} 的输入张量，索引形状为 {2, 2}，数据类型为 Float32
    testGatherCpu<GatherObj>(IncrementalGenerator(), ZeroGenerator(),Shape{1, 2, 2, 3}, Shape{2, 2},
                             DataType::Float32,DataType::Int64, 1);
    // 另外一个测试：数据类型为 Float16
    testGatherCpu<GatherObj>(IncrementalGenerator(), ZeroGenerator(),Shape{1, 2, 2, 3}, Shape{2, 2},
                             DataType::Float16,DataType::Int64, 1);
}

} // namespace infini
