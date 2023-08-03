#include "bang/bang_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

template <class T>
void testUnaryKernel(const std::function<void(void *, size_t, DataType)> &generator,
              const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto bangRuntime = make_ref<BangRuntimeObj>();

    // Notifier and queue
    cnrtNotifier_t st, et;
    CNRT_CHECK(cnrtNotifierCreate(&st));
    CNRT_CHECK(cnrtNotifierCreate(&et));
    auto handle = bangRuntime->cnnlHandle();
    cnrtQueue_t queue;
    cnnlGetQueue(handle, &queue);

    // Build input data on CPU
    Tensor inputCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    inputCpu->dataMalloc();
    inputCpu->setData(generator);

    // GPU
    Graph bangGraph = make_ref<GraphObj>(bangRuntime);
    auto inputGpu = bangGraph->cloneTensor(inputCpu);
    std::vector<int> op_list = {3,2,3};

    auto gpuOp = bangGraph->addOp<T>(inputGpu, nullptr, op_list);

    bangGraph->dataMalloc();
    CNRT_CHECK(cnrtPlaceNotifier(st, queue));
    bangRuntime->run(bangGraph);
    CNRT_CHECK(cnrtPlaceNotifier(et, queue));
    CNRT_CHECK(cnrtQueueSync(queue));
    float latency;
    CNRT_CHECK(cnrtNotifierDuration(st, et, &latency));
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    printf("单目融合 Kernel Hardware Time：%.3f us\n", latency);
    EXPECT_TRUE(1);
}

void testUnaryNofusion(const std::function<void(void *, size_t, DataType)> &generator,
              const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto bangRuntime = make_ref<BangRuntimeObj>();

    // Notifier and queue
    cnrtNotifier_t st, et;
    CNRT_CHECK(cnrtNotifierCreate(&st));
    CNRT_CHECK(cnrtNotifierCreate(&et));
    auto handle = bangRuntime->cnnlHandle();
    cnrtQueue_t queue;
    cnnlGetQueue(handle, &queue);

    // Build input data on CPU
    Tensor inputCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    inputCpu->dataMalloc();
    inputCpu->setData(generator);

    // GPU
    Graph bangGraph = make_ref<GraphObj>(bangRuntime);
    auto inputGpu = bangGraph->cloneTensor(inputCpu);

    auto gpuOp = bangGraph->addOp<SigmoidObj>(inputGpu, nullptr);
    auto outputGpu = gpuOp->getOutput();
    auto gpuOp2 = bangGraph->addOp<ReluObj>(outputGpu, nullptr);
    auto outputGpu2 = gpuOp2->getOutput();
    auto gpuOp3 = bangGraph->addOp<SigmoidObj>(outputGpu2, nullptr);

    bangGraph->dataMalloc();
    CNRT_CHECK(cnrtPlaceNotifier(st, queue));
    bangRuntime->run(bangGraph);
    CNRT_CHECK(cnrtPlaceNotifier(et, queue));
    CNRT_CHECK(cnrtQueueSync(queue));
    float latency;
    CNRT_CHECK(cnrtNotifierDuration(st, et, &latency));
    auto outputGpu3 = gpuOp3->getOutput();
    auto outputGpu2Cpu = outputGpu3->clone(cpuRuntime);
    printf("单目不融合 Kernel Hardware Time：%.3f us\n", latency);
    EXPECT_TRUE(1);
}

TEST(cnnl_unary_kernel, run) {
    testUnaryKernel<UnaryKernelObj>(IncrementalGenerator(), Shape{1024, 1024, 1, 1});
    testUnaryNofusion(IncrementalGenerator(), Shape{1024, 1024, 1, 1});
}

} // namespace infini
