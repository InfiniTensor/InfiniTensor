#include "core/graph.h"
#include "core/runtime.h"
#include "operators/reduce.h"
#include "test.h"
namespace infini {

template <class T>
void testReduceCpu(
    const std::function<void(void*, size_t, DataType)> &generator,std::optional<std::vector<int>> &axis,
    const Shape &shape,const DataType &dataType,bool keepDims
){
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto x = g->addTensor(shape, dataType);  // x 张量
    // auto axis=g->addTensor(axes_shape,DataType::Float32);
    auto _keepDims=keepDims;
    auto op = g->addOp<T>(x, nullptr,axis,_keepDims);
    // if(reduce_type==0){
    //     auto op = g->addOp<ReduceMaxObj>(x, nullptr,axes,_keepDims);
    // }else if(reduce_type==1){
    //     auto op = g->addOp<ReduceMeanObj>(x, nullptr,axes,_keepDims);
    // }else{
    //     auto op = g->addOp<ReduceMinObj>(x, nullptr,axes,_keepDims);
    // }
    g->dataMalloc();
    x->setData(generator);  // 填充 condition 数据


    op->getOutput()->print();
    op->getOutput()->printData();

    runtime->run(g);
    EXPECT_TRUE(1); // 替换为实际的断言
   
}


// 测试用例：形状为 {1, 2, 2, 3} 的张量，数据类型为 Float32
TEST(ElementWise, Cpu) {
    std::optional<std::vector<int>> axes = std::vector<int>{1};  // 传递 optional 类型

    testReduceCpu<ReduceMaxObj>(
        IncrementalGenerator(),            // 输入数据递增
        axes,                    // reduce axis 设置为1
        Shape{1, 2, 2, 3},                 // 输入 shape
 
        DataType::Float32,
        true                               // keepDims
    );
    testReduceCpu<ReduceMeanObj>(
        IncrementalGenerator(),
        axes,
        Shape{1, 2, 2, 3},

        DataType::Float32,
        false
    );
    testReduceCpu<ReduceMinObj>(
        IncrementalGenerator(),
        axes,
        Shape{1, 2, 2, 3},

        DataType::Float32,
        true
    );
}
    
    


}