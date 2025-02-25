#include "core/graph.h"
#include "core/runtime.h"

#include "operators/where.h"

#include "test.h"

namespace infini {
    void testWhereCpu(
        const std::function<void(void *, size_t, DataType)> &generator_inputx,
        const std::function<void(void *, size_t, DataType)> &generator_inputy,
        const std::function<void(void *, size_t, DataType)> &generator_condition,
        const std::function<void(void *, size_t, DataType)> &generator_output,
        const Shape &shape_inputx,
        const Shape &shape_inputy,
        const Shape &shape_condition,
        const DataType &dataType
    ){
        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        Graph g = make_ref<GraphObj>(runtime);
        auto inputx = g->addTensor(shape_inputx, dataType);
        auto inputy = g->addTensor(shape_inputy, dataType);
        auto condition = g->addTensor(shape_condition, DataType::UInt8);
    
        auto op = g->addOp<WhereObj>(inputx, inputy, condition, nullptr);
        g->dataMalloc();
        inputx->setData(generator_inputx);
        inputy->setData(generator_inputy);
        condition->setData(generator_condition);
        
        runtime->run(g);
        EXPECT_TRUE(1);
    }
TEST(Where, Cpu_f32) {
    testWhereCpu(
        IncrementalGenerator(),
        ValGenerator<1>(),
        [](void *data, size_t size, DataType dtype) {  
            auto ptr = static_cast<uint8_t *>(data);
            for (size_t i = 0; i < size; ++i) {
                ptr[i] = static_cast<uint8_t>(i % 2);
            }
        },
        IncrementalGenerator(),
        {3, 3, 4},       
        {3, 3, 4},       
        {3, 3, 4},      
        DataType::Float16);
    testWhereCpu(
        [](void *data, size_t size, DataType dtype) {  
            auto ptr = static_cast<float *>(data);
            for (size_t i = 0; i < size; ++i) {
                ptr[i] = i;
            }
        },
        [](void *data, size_t size, DataType dtype) {  
            auto ptr = static_cast<float *>(data);
            for (size_t i = 0; i < size; ++i) {
                ptr[i] = i * 2;
            }
        },
        [](void *data, size_t size, DataType dtype) {  
            auto ptr = static_cast<uint8_t *>(data);
            for (size_t i = 0; i < size; ++i) {
                ptr[i] = static_cast<uint8_t>(i % 2 == 0);
            }
        },
        IncrementalGenerator(),
        {2, 1, 3},       
        {1, 4, 3},       
        {2, 4, 1},      
        DataType::Float32);
    testWhereCpu(
        [](void *data, size_t size, DataType dtype) {  
            auto ptr = static_cast<float *>(data);
            for (size_t i = 0; i < size; ++i) {
                ptr[i] = i;
            }
        },
        [](void *data, size_t size, DataType dtype) {  
            auto ptr = static_cast<float *>(data);
            for (size_t i = 0; i < size; ++i) {
                ptr[i] = i * 2;
            }
        },
        [](void *data, size_t size, DataType dtype) {  
            auto ptr = static_cast<uint8_t *>(data);
            for (size_t i = 0; i < size; ++i) {
                ptr[i] = static_cast<uint8_t>(i % 2);
            }
        },
        IncrementalGenerator(),
        {3, 1},       
        {3, 4},       
        {1, 4},      
        DataType::Float32);
    }
}
