#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/attention_kvcache.h"

#include "test.h"

namespace infini {
TEST(AttentionKVCache64, Cuda) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();

    Graph gCpu = make_ref<GraphObj>(runtime);

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);
    auto input_k_cache_d = gCuda->addTensor({1, 1, 1, 64}, DataType::Float32);
    auto input_v_cache_d = gCuda->addTensor({1, 1, 1, 64}, DataType::Float32);
    auto input_q_d = gCuda->addTensor({1, 1, 1, 64}, DataType::Float32);
    auto input_k_d = gCuda->addTensor({1, 1, 1, 64}, DataType::Float32);
    auto input_v_d = gCuda->addTensor({1, 1, 1, 64}, DataType::Float32);
    auto position_id_d = gCuda->addTensor({1, 1}, DataType::UInt32);

    auto op = gCuda->addOp<AttentionKVCacheObj>(
        input_k_cache_d, input_v_cache_d, input_q_d, input_k_d, input_v_d,
        position_id_d, nullptr, nullptr, nullptr);
    gCuda->dataMalloc();

    input_q_d->setData(OneGenerator());
    input_k_d->setData(OneGenerator());
    input_v_d->setData(OneGenerator());
    position_id_d->setData(IncrementalGenerator());
    cudaRuntime->run(gCuda);

    auto oCpu = gCpu->cloneTensor(op->getOutputs()[0]);
    EXPECT_TRUE(oCpu->equalData(vector<float>{
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
}

TEST(AttentionKVCache128, Cuda) {
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
        position_id_d, nullptr, nullptr, nullptr);
    gCuda->dataMalloc();

    input_q_d->setData(OneGenerator());
    input_k_d->setData(OneGenerator());
    input_v_d->setData(OneGenerator());
    position_id_d->setData(IncrementalGenerator());
    cudaRuntime->run(gCuda);

    auto oCpu = gCpu->cloneTensor(op->getOutputs()[0]);
    EXPECT_TRUE(oCpu->equalData(vector<float>{
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
}

} // namespace infini
