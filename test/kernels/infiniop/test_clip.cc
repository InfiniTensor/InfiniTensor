#include "core/graph.h"
#include "core/runtime.h"
#ifdef USE_CUDA
#include "cuda/cuda_runtime.h"
#endif
#include "operators/unary.h"

#include "test.h"

namespace infini {

template <class T>
void testClipCpu(const std::function<void(void *, size_t, DataType)> &generator,
                 const Shape &x_shape, const DataType &dataType,
                 std::optional<float> min = std::nullopt,
                 std::optional<float> max = std::nullopt) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto x = g->addTensor(x_shape, dataType);

    auto op = g->addOp<T>(x, nullptr, min, max);
    g->dataMalloc();
    x->setData(generator);

    runtime->run(g);
    // op->getOutput()->print();
    // op->getOutput()->printData();
    EXPECT_TRUE(1);
}

#ifdef USE_CUDA
template <class T,
          typename std::enable_if<std::is_base_of<ClipObj, T>{}, int>::type = 0>
void testClipCuda(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &x_shape, const DataType &dataType,
    std::optional<float> min = std::nullopt,
    std::optional<float> max = std::nullopt) {
    // cpu
    auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Graph cpuG = make_ref<GraphObj>(cpuRuntime);
    auto cpux = cpuG->addTensor(x_shape, dataType);

    auto cpuOp = cpuG->addOp<T>(cpux, nullptr, min, max);
    cpuG->dataMalloc();
    cpux->setData(generator);

    cpuRuntime->run(cpuG);
    auto cpuOutput = cpuOp->getOutput();

    // cuda
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    Graph cudaG = make_ref<GraphObj>(cudaRuntime);
    auto cudax = cudaG->addTensor(x_shape, dataType);
    auto cudaOp = cudaG->addOp<T>(cudax, nullptr, min, max);
    cudaG->dataMalloc();
    cudax->setData(generator);

    cudaRuntime->run(cudaG);
    auto cudaOutput = cudaOp->getOutput()->clone(cpuRuntime);

    EXPECT_TRUE(cudaOutput->equalData(cpuOutput));
}
#endif

TEST(Clip, Cpu) {
    testClipCpu<ClipObj>(IncrementalGenerator(), Shape{3, 2},
                         DataType::Float32);
    testClipCpu<ClipObj>(IncrementalGenerator(), Shape{3, 2},
                         DataType::Float16);
    testClipCpu<ClipObj>(IncrementalGenerator(), Shape{3, 2}, DataType::Float32,
                         0.0f, 1.0f);
    testClipCpu<ClipObj>(IncrementalGenerator(), Shape{3, 2}, DataType::Float16,
                         0.0f, 1.0f);
    testClipCpu<ClipObj>(IncrementalGenerator(), Shape{3, 2}, DataType::Float32,
                         0.0f, std::nullopt);
    testClipCpu<ClipObj>(IncrementalGenerator(), Shape{3, 2}, DataType::Float16,
                         0.0f, std::nullopt);
    testClipCpu<ClipObj>(IncrementalGenerator(), Shape{3, 2}, DataType::Float32,
                         std::nullopt, 1.0f);
    testClipCpu<ClipObj>(IncrementalGenerator(), Shape{3, 2}, DataType::Float16,
                         std::nullopt, 1.0f);
}

#ifdef USE_CUDA
TEST(Clip, Cuda) {
    testClipCuda<ClipObj>(IncrementalGenerator(), Shape{3, 2},
                          DataType::Float32);
    testClipCuda<ClipObj>(IncrementalGenerator(), Shape{3, 2},
                          DataType::Float16);
    testClipCuda<ClipObj>(IncrementalGenerator(), Shape{3, 2},
                          DataType::Float32, 0.0f, 1.0f);
    testClipCuda<ClipObj>(IncrementalGenerator(), Shape{3, 2},
                          DataType::Float16, 0.0f, 1.0f);
    testClipCuda<ClipObj>(IncrementalGenerator(), Shape{3, 2},
                          DataType::Float32, 0.0f, std::nullopt);
    testClipCuda<ClipObj>(IncrementalGenerator(), Shape{3, 2},
                          DataType::Float16, 0.0f, std::nullopt);
    testClipCuda<ClipObj>(IncrementalGenerator(), Shape{3, 2},
                          DataType::Float32, std::nullopt, 1.0f);
    testClipCuda<ClipObj>(IncrementalGenerator(), Shape{3, 2},
                          DataType::Float16, std::nullopt, 1.0f);
}
#endif

} // namespace infini
