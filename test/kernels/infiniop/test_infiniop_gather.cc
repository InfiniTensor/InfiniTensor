#include "core/graph.h"
#include "core/runtime.h"

#include "operators/gather.h"

#include "test.h"

namespace infini {
void testGatherCpu(
    const std::function<void(void *, size_t, DataType)> &generator_input,
    const std::function<void(void *, size_t, DataType)> &generator_output,
    const std::function<void(void *, size_t, DataType)> &generator_indices,
    const int axis,
    const Shape &shape_input,
    const Shape &shape_indices,
    const DataType &dataType
){
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shape_input, dataType);
    auto indices = g->addTensor(shape_indices, DataType::Int64);
    auto op = g->addOp<GatherObj>(input, indices, nullptr, axis);
    auto output = op->getOutput();
    g->dataMalloc();
    input->setData(generator_input);
    indices->setData(generator_indices);
    output->setData(generator_output);
    runtime->run(g);
    input->printData();
    indices->printData();
    output->printData();
    std::cout << "xxxxxxxx" << std::endl;
    EXPECT_TRUE(1);
}
TEST(Gather, Cpu) {
    testGatherCpu(
         
        [](void *data, size_t size, DataType dtype) {
            auto ptr = static_cast<float*>(data);
            for (size_t i = 0; i < size; ++i){
                ptr[i] = i;
            } 
        },
         
        [](void *data, size_t size, DataType dtype) {}, // 空函数
         
        [](void *data, size_t size, DataType dtype) {
            auto ptr = static_cast<int64_t*>(data);

            for (size_t i = 0; i < size; ++i) {
                ptr[i] = static_cast<int64_t>(i % 2); // 生成 0, 1
                }
            },
        0,               
        {3, 4},          
        {2},             
        DataType::Float32
    );
    testGatherCpu(
         
        [](void *data, size_t size, DataType dtype) {
            auto ptr = static_cast<float*>(data);
            for (size_t i = 0; i < size; ++i){
                ptr[i] = i;
            } 
        },
        
        [](void *data, size_t size, DataType dtype) {}, // 空函数
         
        [](void *data, size_t size, DataType dtype) {
            auto ptr = static_cast<int64_t*>(data);

            for (size_t i = 0; i < size; ++i) {
                ptr[i] = static_cast<int64_t>(i % 3); // 生成 0, 1
                }
            },
        1,                
        {3, 4, 4},           
        {3},              
        DataType::Float32
    );
    testGatherCpu(
        IncrementalGenerator(),
        [](void *data, size_t size, DataType dtype) {}, // 空函数
        [](void *data, size_t size, DataType dtype) {
            auto ptr = static_cast<int64_t*>(data);

            for (size_t i = 0; i < size; ++i) {
                ptr[i] = static_cast<int64_t>(i % 2); // 生成 0, 1
                }
            },
        0,               
        {3, 4},          
        {2},             
        DataType::Float16
    );
}
}