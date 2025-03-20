#include "core/graph.h"
#include "core/runtime.h"
#ifdef USE_CUDA
#include "cuda/cuda_runtime.h"
#endif
#include "operators/where.h"

#include "test.h"

namespace infini {
void testWhereCpu(
    const std::function<void(void *, size_t, DataType)> &generator_inputx,
    const std::function<void(void *, size_t, DataType)> &generator_inputy,
    const std::function<void(void *, size_t, DataType)> &generator_condition,
    const Shape &x_shape,
    const Shape &y_shape,
    const Shape &cond_shape,
    const DataType &dataType
){
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto inputx = g->addTensor(x_shape, dataType);
    auto inputy = g->addTensor(y_shape, dataType);
    auto condition = g->addTensor(cond_shape, DataType::UInt8);
    
    auto op = g->addOp<WhereObj>(inputx, inputy, condition, nullptr);
    g->dataMalloc();
    inputx->setData(generator_inputx);
    inputy->setData(generator_inputy);
    condition->setData(generator_condition);
        
    runtime->run(g);
    EXPECT_TRUE(1);
}
#ifdef USE_CUDA
template <class T>
void testWhereCuda(
    const std::function<void(void *, size_t, DataType)> &generator_inputx,
    const std::function<void(void *, size_t, DataType)> &generator_inputy,
    const std::function<void(void *, size_t, DataType)> &generator_condition,
    const Shape &x_shape, const Shape &y_shape, const Shape &cond_shape,
    const DataType &dataType) {
    // cpu
    auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Graph cpuG = make_ref<GraphObj>(cpuRuntime);
    auto cpuX = cpuG->addTensor(x_shape, dataType);
    auto cpuY = cpuG->addTensor(y_shape, dataType);
    auto cpuCond = cpuG->addTensor(cond_shape, DataType::UInt8);

    auto cpuOp = cpuG->addOp<T>(cpuX, cpuY, cpuCond, nullptr);
    cpuG->dataMalloc();
    cpuX->setData(generator_inputx);
    cpuY->setData(generator_inputy);
    cpuCond->setData(generator_condition);

    cpuRuntime->run(cpuG);
    auto cpuOutput = cpuOp->getOutput();

    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    Graph cudaG = make_ref<GraphObj>(cudaRuntime);
    auto cudaX = cudaG->addTensor(x_shape, dataType);
    auto cudaY = cudaG->addTensor(y_shape, dataType);
    auto cudaCond = cudaG->addTensor(cond_shape, DataType::UInt8);
    auto cudaOp = cudaG->addOp<T>(cudaX, cudaY, cudaCond, nullptr);
    cudaG->dataMalloc();
    cudaX->setData(generator_inputx);
    cudaY->setData(generator_inputy);
    cudaCond->setData(generator_condition);

    cudaRuntime->run(cudaG);
    auto cudaOutput = cudaOp->getOutput()->clone(cpuRuntime);

    EXPECT_TRUE(cudaOutput->equalData(cpuOutput));
}
#endif

TEST(Where, Cpu_f32) {
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
        {3, 1},       
        {3, 4},       
        {1, 4},      
        DataType::Float32);
    }
#ifdef USE_CUDA
TEST(Where, Cpu_f32) {
    testWhereCuda(
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
        {2, 1, 3},       
        {1, 4, 3},       
        {2, 4, 1},      
        DataType::Float32);
    testWhereCuda(
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
        {3, 1},       
        {3, 4},       
        {1, 4},      
        DataType::Float32);
    testWhereCuda(
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
        {2, 1, 3},       
        {1, 4, 3},       
        {2, 4, 1},      
        DataType::Float16);
    testWhereCuda(
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
        {3, 1},       
        {3, 4},       
        {1, 4},      
        DataType::Float16);
    }
    

}
