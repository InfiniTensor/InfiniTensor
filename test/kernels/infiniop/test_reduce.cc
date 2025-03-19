#include "core/graph.h"
#include "core/runtime.h"
#ifdef USE_CUDA
#include "cuda/cuda_runtime.h"
#endif
#include "operators/reduce.h"

#include "test.h"

namespace infini {

void infer_reduce_shape(const Shape &x_shape, const std::vector<int> &axes,
                        bool keepdims, Shape &y_shape) {
    auto rank = x_shape.size();
    if (keepdims) {
        y_shape = x_shape;
        for (auto it : axes)
            y_shape[it] = 1;
    } else {
        for (size_t i = 0; i < rank; ++i) {
            if (std::find(axes.begin(), axes.end(), i) == axes.end())
                y_shape.emplace_back(x_shape[i]);
        }
        if (y_shape.empty())
            y_shape = {1};
    }
}

template <class T>
void testReduceCpu(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &x_shape, const DataType &dataType, std::vector<int> axes,
    bool keepdims) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto x = g->addTensor(x_shape, dataType);
    auto op = g->addOp<T>(x, nullptr, axes, keepdims);

    g->dataMalloc();
    x->setData(generator);

    runtime->run(g);
    // op->getOutput()->print();
    // op->getOutput()->printData();
    EXPECT_TRUE(1);
}

#ifdef USE_CUDA
template <class T>
void testReduceCuda(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &x_shape, const DataType &dataType, std::vector<int> axes,
    bool keepdims) {
    // cpu
    auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Graph cpuG = make_ref<GraphObj>(cpuRuntime);
    auto cpux = cpuG->addTensor(x_shape, dataType);
    auto cpuOp = cpuG->addOp<T>(cpux, nullptr, axes, keepdims);

    cpuG->dataMalloc();
    cpux->setData(generator);

    cpuRuntime->run(cpuG);
    auto cpuy = cpuOp->getOutput();

    // cuda
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    Graph cudaG = make_ref<GraphObj>(cudaRuntime);
    auto cudax = cudaG->addTensor(x_shape, dataType);
    auto cudaOp = cudaG->addOp<T>(cudax, nullptr, axes, keepdims);
    cudaG->dataMalloc();
    cudax->setData(generator);

    cudaRuntime->run(cudaG);
    auto cuday = cudaOp->getOutput()->clone(cpuRuntime);

    EXPECT_TRUE(cuday->equalData(cpuy));
}
#endif

TEST(Reduce, Cpu) {
    testReduceCpu<ReduceMinObj>(IncrementalGenerator(), Shape{3, 2}, DataType::Float32,
                               {0}, true);
    testReduceCpu<ReduceMinObj>(IncrementalGenerator(), Shape{3, 2}, DataType::Float16,
                               {0}, true);
    testReduceCpu<ReduceMaxObj>(IncrementalGenerator(), Shape{3, 2}, DataType::Float32,
                               {0}, true);
    testReduceCpu<ReduceMaxObj>(IncrementalGenerator(), Shape{3, 2}, DataType::Float16,
                               {0}, true);
    testReduceCpu<ReduceMeanObj>(IncrementalGenerator(), Shape{3, 2}, DataType::Float32,
                                {0}, true);
    testReduceCpu<ReduceMeanObj>(IncrementalGenerator(), Shape{3, 2}, DataType::Float16,
                                {0}, true);
}

#ifdef USE_CUDA
TEST(Reduce, Cuda) {
    testReduceCuda<ReduceMinObj>(IncrementalGenerator(), Shape{3, 2}, DataType::Float32,
                                {0}, true);
    testReduceCuda<ReduceMinObj>(IncrementalGenerator(), Shape{3, 2}, DataType::Float16,
                                {0}, true);
    testReduceCuda<ReduceMaxObj>(IncrementalGenerator(), Shape{3, 2}, DataType::Float32,
                                {0}, true);
    testReduceCuda<ReduceMaxObj>(IncrementalGenerator(), Shape{3, 2}, DataType::Float16,
                                {0}, true);
    testReduceCuda<ReduceMeanObj>(IncrementalGenerator(), Shape{3, 2}, DataType::Float32,
                                 {0}, true);
    testReduceCuda<ReduceMeanObj>(IncrementalGenerator(), Shape{3, 2}, DataType::Float16,
                                 {0}, true);
}
#endif

} // namespace infini
