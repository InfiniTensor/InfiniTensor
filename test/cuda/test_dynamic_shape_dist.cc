#ifdef INFINI_USE_NCCL
#include "core/graph.h"
#include "cuda/cuda_runtime.h"
#include "cuda/nccl_communicator.h"
#include "operators/concat.h"
#include "operators/gather.h"
#include "operators/reshape.h"
#include "operators/unary.h"
#include "test.h"
#include <thread>

namespace infini {
namespace {

constexpr int kWorldSize = 2;
constexpr size_t kWorkspaceSize = 64ull << 20;

void runRank(int rank, float *results) {
    auto runtime = make_ref<CudaRuntimeObj>(rank, 4, kWorkspaceSize);
    runtime->initComm("dynamic_shape_dist", kWorldSize, rank);

    auto graph = make_ref<GraphObj>(runtime);
    const int batch = 2 + rank;
    auto input = graph->addTensor({batch, 2, 3}, DataType::Float32);
    auto index = graph->addTensor({1}, DataType::Int64);
    auto tail = graph->addTensor({1}, DataType::Int64);
    input->setInput();
    index->setInput();
    tail->setInput();
    auto shape = graph->addOp<ShapeObj>(input, nullptr)->getOutput();
    auto selected = graph->addOp<GatherObj>(shape, index, nullptr, 0)->getOutput();
    auto target = graph->addOp<ConcatObj>(TensorVec{selected, tail}, nullptr, 0)
                      ->getOutput();
    auto output = graph->addOp<ReshapeObj>(input, target, nullptr)->getOutput();
    output->setOutput();

    graph->dataMalloc();
    index->copyin<int64_t>({0});
    tail->copyin<int64_t>({6});
    graph->prepareDynamicShapes();
    graph->shape_infer();
    graph->dataMalloc();

    std::vector<float> values(static_cast<size_t>(batch) * 6, 1.0f);
    input->copyin(values);
    runtime->run(graph);
    const Shape expectedShape{batch, 6};
    IT_ASSERT(output->getDims() == expectedShape);

    float *deviceResult = static_cast<float *>(runtime->alloc(sizeof(float)));
    const float localElements = static_cast<float>(output->size());
    runtime->copyBlobFromCPU(deviceResult, &localElements, sizeof(float));
    auto comm = dynamic_cast<NcclCommunicatorObj &>(runtime->getCommunicator())
                    .getNcclComm();
    checkNcclError(ncclAllReduce(deviceResult, deviceResult, 1, ncclFloat,
                                 ncclSum, comm, CUDAStream::getCurrentStream()));
    runtime->sync();
    runtime->copyBlobToCPU(results + rank, deviceResult, sizeof(float));
    runtime->dealloc(deviceResult);
}

} // namespace

TEST(NCCL, dynamic_shape_runs_on_both_gpus) {
    float results[kWorldSize] = {};
    std::thread threads[kWorldSize];
    for (int rank = 0; rank < kWorldSize; ++rank)
        threads[rank] = std::thread(runRank, rank, results);
    for (auto &thread : threads)
        thread.join();

    EXPECT_FLOAT_EQ(results[0], 30.0f);
    EXPECT_FLOAT_EQ(results[1], 30.0f);
}

} // namespace infini
#endif
