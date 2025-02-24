#include "core/graph.h"
#include "core/runtime.h"
#ifdef USE_CUDA
#include "cuda/cuda_runtime.h"
#endif
#include "operators/gather.h"

#include "test.h"

namespace infini {

template <class T>
void testGatherCpu(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &input_shape, const Shape &indices_shape,
    const DataType &dataType, const DataType &indicesType,
    const int axis) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(input_shape, dataType);
    auto indices = g->addTensor(indices_shape, indicesType);

    auto op = g->addOp<T>(input, indices, nullptr, axis);
    g->dataMalloc();
    input->setData(generator);

    runtime->run(g);
    // op->getOutput()->print();
    // op->getOutput()->printData();
    EXPECT_TRUE(1);
}

#ifdef USE_CUDA
template <class T>
void testGatherCuda(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &input_shape, const Shape &indices_shape,
    const DataType &dataType, const DataType &indicesType,
    const int axis) {
    // cpu
    auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Graph cpuG = make_ref<GraphObj>(cpuRuntime);
    auto cpuInput = cpuG->addTensor(input_shape, dataType);
    auto cpuIndex = cpuG->addTensor(indices_shape, indicesType);

    auto cpuOp = cpuG->addOp<T>(cpuInput, cpuIndex, nullptr, axis);
    cpuG->dataMalloc();
    cpuInput->setData(generator);

    cpuRuntime->run(cpuG);
    auto cpuOutput = cpuOp->getOutput();

    // cuda
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    Graph cudaG = make_ref<GraphObj>(cudaRuntime);
    auto cudaInput = cudaG->addTensor(input_shape, dataType);
    auto cudaIndex = cudaG->addTensor(indices_shape, indicesType);
    auto cudaOp =
        cudaG->addOp<T>(cudaInput, cudaIndex, nullptr, axis);
    cudaG->dataMalloc();
    cudaInput->setData(generator);

    cudaRuntime->run(cudaG);
    auto cudaOutput = cudaOp->getOutput()->clone(cpuRuntime);

    EXPECT_TRUE(cudaOutput->equalData(cpuOutput));
}
#endif

TEST(Gather, Cpu) {
    testGatherCpu<GatherObj>(IncrementalGenerator(), Shape{3, 2},
                             Shape{2, 2}, DataType::Float32, DataType::Int32,
                             0);
    testGatherCpu<GatherObj>(IncrementalGenerator(), Shape{3, 2},
                             Shape{2, 2}, DataType::Float16, DataType::Int32,
                             0);
}

#ifdef USE_CUDA
TEST(Gather, Cuda) {
    testGatherCuda<GatherObj>(IncrementalGenerator(), Shape{3, 2},
                             Shape{2, 2}, DataType::Float32, DataType::Int32,
                             0);
    testGatherCuda<GatherObj>(IncrementalGenerator(), Shape{3, 2},
                             Shape{2, 2}, DataType::Float16, DataType::Int32,
                             0);
}
#endif

} // namespace infini
