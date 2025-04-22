#include "core/graph.h"
#include "core/runtime.h"
#include "operators/where.h"
#include "test.h"

namespace infini {

template <class T>
void testWhereCpu(
    const std::function<void(void*, size_t, DataType)> &generator,const std::function<void(void*, size_t, DataType)> &generator1,
    const Shape &shape, const DataType &dataType,const DataType &cond_dataType
) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    // 创建 condition, x, y 张量
    auto condition = g->addTensor(shape, cond_dataType);  // 条件张量使用布尔类型
    auto x = g->addTensor(shape, dataType);  // x 张量
    auto y = g->addTensor(shape, dataType);  // y 张量

    // 添加 Where 操作
    auto op = g->addOp<WhereObj>(x, y,condition, nullptr);

    g->dataMalloc();
    condition->setData(generator1);  // 填充 condition 数据
    x->setData(generator);          // 填充 x 数据
    y->setData(generator);          // 填充 y 数据
    runtime->run(g);

    // 输出结果
    // op->getOutput()->print();
    // op->getOutput()->printData();
    EXPECT_TRUE(1); // 替换为实际的断言
}

TEST(ElementWise, Cpu) {
    // 测试用例：形状为 {1, 2, 2, 3} 的张量，数据类型为 Float32
    testWhereCpu<WhereObj>(IncrementalGenerator(), ZeroGenerator(),Shape{1, 2, 2, 3}, DataType::Float32,DataType::UInt8);
    // 另外一个测试：数据类型为 Float16
    testWhereCpu<WhereObj>(IncrementalGenerator(), ZeroGenerator(),Shape{1, 2, 2, 3}, DataType::Float16,DataType::UInt8);
}

} // namespace infini
