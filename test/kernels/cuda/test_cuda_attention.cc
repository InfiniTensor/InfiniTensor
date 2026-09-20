#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/attention_kvcache.h"

#include "test.h"

namespace infini {
TEST(AttentionKVCache, Cuda) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();

    Graph gCpu = make_ref<GraphObj>(runtime);

    // A small workspace must suffice even across multiple 16-token tiles.
    auto cudaRuntime = make_ref<CudaRuntimeObj>(0, 16, 1 << 20);
    for (int length : {1, 17, 33}) {
        SCOPED_TRACE(length);
        Graph gCuda = make_ref<GraphObj>(cudaRuntime);
        auto input_k_cache_d =
            gCuda->addTensor({1, 1, length, 128}, DataType::Float32);
        auto input_v_cache_d =
            gCuda->addTensor({1, 1, length, 128}, DataType::Float32);
        auto input_q_d = gCuda->addTensor({1, 1, 1, 128}, DataType::Float32);
        auto input_k_d = gCuda->addTensor({1, 1, 1, 128}, DataType::Float32);
        auto input_v_d = gCuda->addTensor({1, 1, 1, 128}, DataType::Float32);
        auto position_id_d = gCuda->addTensor({1, 1}, DataType::UInt32);

        auto op = gCuda->addOp<AttentionKVCacheObj>(
            input_k_cache_d, input_v_cache_d, input_q_d, input_k_d, input_v_d,
            position_id_d, nullptr);
        gCuda->dataMalloc();

        input_q_d->setData(OneGenerator());
        input_k_cache_d->setData(OneGenerator());
        input_v_cache_d->setData(OneGenerator());
        input_k_d->setData(OneGenerator());
        input_v_d->setData(OneGenerator());
        position_id_d->copyin(
            vector<uint32_t>{static_cast<uint32_t>(length - 1)});
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
}

} // namespace infini
