#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/attention_kvcache.h"

#include "test.h"

namespace infini {

TEST(TestCudaRuntime, CudaGraph) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();

    Graph gCpu = make_ref<GraphObj>(runtime);

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto input_k_cache_d = gCuda->addTensor({1, 1, 1, 128}, DataType::Float32);
    auto input_v_cache_d = gCuda->addTensor({1, 1, 1, 128}, DataType::Float32);
    auto input_q_d = gCuda->addTensor({1, 1, 1, 128}, DataType::Float32);
    auto input_k_d = gCuda->addTensor({1, 1, 1, 128}, DataType::Float32);
    auto input_v_d = gCuda->addTensor({1, 1, 1, 128}, DataType::Float32);
    auto position_id_d = gCuda->addTensor({1, 1}, DataType::UInt32);

    auto op = gCuda->addOp<AttentionKVCacheObj>(
        input_k_cache_d, input_v_cache_d, input_q_d, input_k_d, input_v_d,
        position_id_d, nullptr);
    auto op1 = gCuda->addOp<AttentionKVCacheObj>(
        input_k_cache_d, input_v_cache_d, op->getOutputs()[0], input_k_d,
        input_v_d, position_id_d, nullptr);
    auto op2 = gCuda->addOp<AttentionKVCacheObj>(
        input_k_cache_d, input_v_cache_d, op1->getOutputs()[0], input_k_d,
        input_v_d, position_id_d, nullptr);
    gCuda->dataMalloc();

    input_q_d->setData(OneGenerator());
    input_k_d->setData(OneGenerator());
    input_v_d->setData(OneGenerator());
    position_id_d->setData(IncrementalGenerator());

    cudaRuntime->run(gCuda);

    cudaEvent_t start, stop;
    float milliseconds_1 = 0, milliseconds_2 = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();
    cudaEventRecord(start);
    cudaRuntime->run(gCuda);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds_1, start, stop);
    printf("without cudaGraph, latency: %f ms\n", milliseconds_1);

    cudaRuntime->runWithCudaGraph(gCuda);
    cudaRuntime->runWithCudaGraph(gCuda);

    cudaDeviceSynchronize();
    cudaEventRecord(start);
    cudaRuntime->runWithCudaGraph(gCuda);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds_2, start, stop);
    printf("with cudaGraph, latency: %f ms\n", milliseconds_2);
    EXPECT_GE(milliseconds_1, milliseconds_2);
}

} // namespace infini
