#include "core/blob.h"
#include "core/graph.h"
#include "core/runtime.h"
#include "operators/element_wise.h"
#include "operators/matmul.h"
#include "operators/unary.h"
#include "test.h"
#include <cstdlib>
#include <new>
#include <unordered_set>

namespace infini {

class TrackingCpuRuntimeObj final : public CpuRuntimeObj {
  private:
    std::unordered_set<void *> livePointers;
    bool failNextAllocation = false;
    bool failNextDeallocation = false;
    mutable bool failNextCopy = false;
    size_t allocationCount = 0;
    size_t deallocationCount = 0;
    vector<size_t> allocationSizes;

  public:
    TrackingCpuRuntimeObj() : CpuRuntimeObj(Device::CPU) {}

    void *alloc(size_t size) override {
        if (failNextAllocation) {
            failNextAllocation = false;
            throw std::bad_alloc();
        }
        auto ptr = std::calloc(1, size);
        if (ptr == nullptr)
            throw std::bad_alloc();
        IT_ASSERT(livePointers.insert(ptr).second,
                  "Allocator returned a live pointer");
        ++allocationCount;
        allocationSizes.emplace_back(size);
        return ptr;
    }

    void dealloc(void *ptr) override {
        IT_ASSERT(ptr != nullptr, "Cannot deallocate a null pointer");
        IT_ASSERT(livePointers.erase(ptr) == 1,
                  "Pointer was not allocated or was already freed");
        std::free(ptr);
        ++deallocationCount;
        if (failNextDeallocation) {
            failNextDeallocation = false;
            throw Exception("Injected deallocation failure");
        }
    }

    string toString() const override { return "Tracking CPU Runtime"; }

    void copyBlobInsideRuntime(void *dst, const void *src,
                               size_t bytes) const override {
        if (failNextCopy) {
            failNextCopy = false;
            IT_ASSERT(false, "Injected copy failure");
        }
        CpuRuntimeObj::copyBlobInsideRuntime(dst, src, bytes);
    }

    void failNextAlloc() { failNextAllocation = true; }
    void failNextDealloc() { failNextDeallocation = true; }
    void failNextBlobCopy() { failNextCopy = true; }
    size_t getLiveAllocationCount() const { return livePointers.size(); }
    size_t getAllocationCount() const { return allocationCount; }
    size_t getDeallocationCount() const { return deallocationCount; }
    const vector<size_t> &getAllocationSizes() const { return allocationSizes; }
    void clearAllocationSizes() { allocationSizes.clear(); }
};

TEST(Graph, build_and_run) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i0 = g->addTensor({1, 2, 3}, DataType::UInt32);
    Tensor w0 = g->addTensor({1, 3, 4}, DataType::UInt32);
    Tensor o0 = g->addTensor({1, 2, 4}, DataType::UInt32);
    g->dataMalloc();
    i0->copyin(vector<uint32_t>{1, 2, 3, 4, 5, 6});
    w0->copyin(vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto matmul = g->addOpWithOutputs<MatmulObj>(i0, w0, o0);
    g->print();
    // check targets and source for tensor
    EXPECT_EQ(i0->getTargets().size(), 1u);
    EXPECT_EQ(w0->getTargets().size(), 1u);
    EXPECT_EQ(o0->getTargets().size(), 0u);
    EXPECT_EQ(i0->getSource(), nullptr);
    EXPECT_EQ(w0->getSource(), nullptr);
    EXPECT_NE(o0->getSource(), nullptr);
    EXPECT_EQ(matmul->getPredecessors().size(), 0u);
    EXPECT_EQ(matmul->getSuccessors().size(), 0u);

    runtime->run(g);
    // check execution results
    auto ans = make_ref<TensorObj>(Shape{1, 2, 4}, DataType::UInt32, runtime);
    ans->dataMalloc();
    ans->copyin(vector<uint32_t>{38, 44, 50, 56, 83, 98, 113, 128});
    EXPECT_TRUE(o0->equalData(ans));
}

TEST(Graph, blob_views_preserve_storage_identity) {
    auto runtime = make_ref<TrackingCpuRuntimeObj>();
    {
        auto storage = runtime->allocBlob(64);
        auto view = make_ref<BlobObj>(storage, 16, 32);
        auto nestedView = make_ref<BlobObj>(view, 8, 8);
        auto otherStorage = runtime->allocBlob(64);

        EXPECT_EQ(view->getStorageId(), storage->getStorageId());
        EXPECT_EQ(view->getStorageOffset(), 16u);
        EXPECT_EQ(nestedView->getStorageId(), storage->getStorageId());
        EXPECT_EQ(nestedView->getStorageOffset(), 24u);
        EXPECT_NE(otherStorage->getStorageId(), storage->getStorageId());
    }
    EXPECT_EQ(runtime->getLiveAllocationCount(), 0u);
}

TEST(Graph, capture_generation_tracks_execution_state) {
    auto runtime = make_ref<TrackingCpuRuntimeObj>();
    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor({4}, DataType::Float32);
    input->setInput();
    g->dataMalloc();

    const auto stableGeneration = g->getCaptureGeneration();
    const auto stableTopology = g->getTopologyEpoch();
    input->setShape({4});
    input->copyin(vector<float>{1, 2, 3, 4});
    EXPECT_EQ(g->getCaptureGeneration(), stableGeneration);
    EXPECT_EQ(g->getTopologyEpoch(), stableTopology);

    input->setShape({2, 2});
    EXPECT_GT(g->getCaptureGeneration(), stableGeneration);
    EXPECT_EQ(g->getTopologyEpoch(), stableTopology);

    const auto layoutGeneration = g->getCaptureGeneration();
    auto unallocated = g->addTensor({1}, DataType::Float32);
    EXPECT_GT(g->getCaptureGeneration(), layoutGeneration);
    EXPECT_GT(g->getTopologyEpoch(), stableTopology);

    const auto generationBeforeFailure = g->getCaptureGeneration();
    runtime->failNextAlloc();
    EXPECT_THROW(unallocated->dataMalloc(), std::bad_alloc);
    EXPECT_EQ(g->getCaptureGeneration(), generationBeforeFailure);
}

TEST(Graph, shared_tensor_notifies_each_graph_with_weak_observers) {
    auto runtime = make_ref<TrackingCpuRuntimeObj>();
    auto tensor = make_ref<TensorObj>(Shape{2}, DataType::Float32, runtime);
    Graph first = make_ref<GraphObj>(runtime);
    Graph second = make_ref<GraphObj>(runtime);
    first->addTensor(tensor);
    second->addTensor(tensor);
    const auto firstGeneration = first->getCaptureGeneration();
    const auto secondGeneration = second->getCaptureGeneration();

    tensor->setShape({1, 2});
    EXPECT_GT(first->getCaptureGeneration(), firstGeneration);
    EXPECT_GT(second->getCaptureGeneration(), secondGeneration);

    first.reset();
    const auto remainingGeneration = second->getCaptureGeneration();
    tensor->setShape({2, 1});
    EXPECT_GT(second->getCaptureGeneration(), remainingGeneration);
}

TEST(Graph, naive_allocator_reallocates_dynamic_tensors) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor input = g->addTensor({1, 2}, DataType::Float32);
    Tensor weight = g->addTensor({2, 2}, DataType::Float32);
    input->setInput();
    weight->setWeight();
    auto matmul = g->addOp<MatmulObj>(input, weight, nullptr);
    Tensor output = matmul->getOutput();
    output->setOutput();

    g->dataMalloc(true);
    auto initialInputBlob = input->getDataBlob();
    auto initialOutputBlob = output->getDataBlob();
    auto weightBlob = weight->getDataBlob();
    weight->copyin(vector<float>{1, 0, 0, 1});

    constexpr int expandedBatch = 8192;
    input->setShape({expandedBatch, 2});
    g->shape_infer();
    g->dataMalloc(true);

    ASSERT_NE(input->getDataBlob(), initialInputBlob);
    ASSERT_NE(output->getDataBlob(), initialOutputBlob);
    EXPECT_EQ(weight->getDataBlob(), weightBlob);
    EXPECT_GE(input->getDataBlob()->getBytes(), input->getBytes());
    EXPECT_GE(output->getDataBlob()->getBytes(), output->getBytes());

    vector<float> expandedInput(expandedBatch * 2);
    for (size_t i = 0; i < expandedInput.size(); ++i)
        expandedInput[i] = static_cast<float>(i);
    input->copyin(expandedInput);
    runtime->run(g);
    EXPECT_TRUE(output->equalData(expandedInput));

    auto expandedInputBlob = input->getDataBlob();
    auto expandedOutputBlob = output->getDataBlob();
    input->setShape({3, 2});
    g->shape_infer();
    g->dataMalloc(true);

    EXPECT_NE(input->getDataBlob(), expandedInputBlob);
    EXPECT_NE(output->getDataBlob(), expandedOutputBlob);
    EXPECT_EQ(weight->getDataBlob(), weightBlob);
    EXPECT_EQ(input->getDataBlob()->getBytes(), input->getBytes());
    EXPECT_EQ(output->getDataBlob()->getBytes(), output->getBytes());
    input->copyin(vector<float>{1, 2, 3, 4, 5, 6});
    runtime->run(g);
    EXPECT_TRUE(output->equalData(vector<float>{1, 2, 3, 4, 5, 6}));

    input->setShape({expandedBatch * 2, 2});
    g->shape_infer();
    g->dataMalloc(true);

    EXPECT_NE(input->getDataBlob(), expandedInputBlob);
    EXPECT_NE(output->getDataBlob(), expandedOutputBlob);
    EXPECT_EQ(weight->getDataBlob(), weightBlob);
    EXPECT_GE(input->getDataBlob()->getBytes(), input->getBytes());
    EXPECT_GE(output->getDataBlob()->getBytes(), output->getBytes());
}

TEST(Graph, lazy_allocator_preserves_preloaded_user_data) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor input = g->addTensor({1, 2}, DataType::Float32);
    Tensor weight = g->addTensor({2, 2}, DataType::Float32);
    input->setInput();
    weight->setWeight();

    input->dataMalloc();
    input->copyin(vector<float>{3, 4});
    weight->dataMalloc();
    weight->copyin(vector<float>{1, 0, 0, 1});

    auto matmul = g->addOp<MatmulObj>(input, weight, nullptr);
    matmul->getOutput()->setOutput();
    g->dataMalloc();
    runtime->run(g);

    EXPECT_TRUE(matmul->getOutput()->equalData(vector<float>{3, 4}));
}

TEST(Graph, lazy_allocator_reuses_high_watermark_capacity) {
    auto runtime = make_ref<TrackingCpuRuntimeObj>();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor input = g->addTensor({8, 2}, DataType::Float32);
        Tensor weight = g->addTensor({2, 2}, DataType::Float32);
        input->setInput();
        weight->setWeight();
        auto matmul = g->addOp<MatmulObj>(input, weight, nullptr);
        Tensor output = matmul->getOutput();
        output->setOutput();

        g->dataMalloc();
        weight->copyin(vector<float>{1, 0, 0, 1});
        const auto allocationCount = runtime->getAllocationCount();
        const auto deallocationCount = runtime->getDeallocationCount();
        const auto inputAddress = input->getRawDataPtr<const void *>();
        const auto initialGeneration = g->getAllocationGeneration();

        g->dataMalloc();
        EXPECT_EQ(runtime->getAllocationCount(), allocationCount);
        EXPECT_EQ(runtime->getDeallocationCount(), deallocationCount);
        EXPECT_EQ(g->getAllocationGeneration(), initialGeneration);

        for (int i = 0; i < 10000; ++i) {
            const int batch = i % 2 == 0 ? 6 : 8;
            input->setShape({batch, 2});
            g->shape_infer();
            g->dataMalloc();
        }

        EXPECT_EQ(runtime->getAllocationCount(), allocationCount);
        EXPECT_EQ(runtime->getDeallocationCount(), deallocationCount);
        EXPECT_EQ(input->getRawDataPtr<const void *>(), inputAddress);

        vector<float> data(16);
        for (size_t i = 0; i < data.size(); ++i)
            data[i] = static_cast<float>(i);
        input->copyin(data);
        runtime->run(g);
        EXPECT_TRUE(output->equalData(data));
    }
    EXPECT_EQ(runtime->getLiveAllocationCount(), 0);
    EXPECT_EQ(runtime->getAllocationCount(), runtime->getDeallocationCount());
}

TEST(Graph, lazy_allocator_grows_and_trims_capacity) {
    auto runtime = make_ref<TrackingCpuRuntimeObj>();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor input = g->addTensor({8, 2}, DataType::Float32);
        Tensor weight = g->addTensor({2, 2}, DataType::Float32);
        input->setInput();
        weight->setWeight();
        auto matmul = g->addOp<MatmulObj>(input, weight, nullptr);
        Tensor output = matmul->getOutput();
        output->setOutput();

        g->dataMalloc();
        weight->copyin(vector<float>{1, 0, 0, 1});
        ASSERT_GE(runtime->getAllocationSizes().size(), 2);
        const auto initialCapacity = runtime->getAllocationSizes().back();
        const auto allocationCount = runtime->getAllocationCount();
        const auto deallocationCount = runtime->getDeallocationCount();

        input->setShape({9, 2});
        g->shape_infer();
        g->dataMalloc();
        ASSERT_EQ(runtime->getAllocationCount(), allocationCount + 1);
        EXPECT_EQ(runtime->getDeallocationCount(), deallocationCount + 1);
        EXPECT_EQ(runtime->getAllocationSizes().back(),
                  initialCapacity + initialCapacity / 2);

        const auto grownAllocationCount = runtime->getAllocationCount();
        const auto grownDeallocationCount = runtime->getDeallocationCount();
        input->setShape({4, 2});
        g->shape_infer();
        g->dataMalloc();
        EXPECT_EQ(runtime->getAllocationCount(), grownAllocationCount);
        EXPECT_EQ(runtime->getDeallocationCount(), grownDeallocationCount);

        vector<float> data{1, 2, 3, 4, 5, 6, 7, 8};
        input->copyin(data);
        const auto generationBeforeTrim = g->getAllocationGeneration();
        g->trimMemory();
        EXPECT_EQ(runtime->getAllocationCount(), grownAllocationCount + 1);
        EXPECT_EQ(runtime->getDeallocationCount(), grownDeallocationCount + 1);
        EXPECT_EQ(runtime->getAllocationSizes().back(), 64);
        EXPECT_GT(g->getAllocationGeneration(), generationBeforeTrim);

        runtime->run(g);
        EXPECT_TRUE(output->equalData(data));

        const auto generationAfterTrim = g->getAllocationGeneration();
        g->trimMemory();
        EXPECT_EQ(runtime->getAllocationCount(), grownAllocationCount + 1);
        EXPECT_EQ(runtime->getDeallocationCount(), grownDeallocationCount + 1);
        EXPECT_EQ(g->getAllocationGeneration(), generationAfterTrim);
    }
    EXPECT_EQ(runtime->getLiveAllocationCount(), 0);
    EXPECT_EQ(runtime->getAllocationCount(), runtime->getDeallocationCount());
}

TEST(Graph, lazy_allocator_trim_failure_preserves_committed_layout) {
    auto runtime = make_ref<TrackingCpuRuntimeObj>();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor input = g->addTensor({8, 2}, DataType::Float32);
        Tensor weight = g->addTensor({2, 2}, DataType::Float32);
        input->setInput();
        weight->setWeight();
        auto matmul = g->addOp<MatmulObj>(input, weight, nullptr);
        Tensor output = matmul->getOutput();
        output->setOutput();

        g->dataMalloc();
        weight->copyin(vector<float>{1, 0, 0, 1});
        input->setShape({4, 2});
        g->shape_infer();
        g->dataMalloc();
        const vector<float> data{1, 2, 3, 4, 5, 6, 7, 8};
        input->copyin(data);

        const auto inputAddress = input->getRawDataPtr<const void *>();
        const auto outputAddress = output->getRawDataPtr<const void *>();
        const auto generation = g->getAllocationGeneration();
        const auto captureGeneration = g->getCaptureGeneration();
        const auto liveAllocations = runtime->getLiveAllocationCount();

        runtime->failNextAlloc();
        EXPECT_THROW(g->trimMemory(), std::bad_alloc);
        EXPECT_EQ(input->getRawDataPtr<const void *>(), inputAddress);
        EXPECT_EQ(output->getRawDataPtr<const void *>(), outputAddress);
        EXPECT_EQ(g->getAllocationGeneration(), generation);
        EXPECT_EQ(g->getCaptureGeneration(), captureGeneration);
        EXPECT_EQ(runtime->getLiveAllocationCount(), liveAllocations);

        runtime->failNextBlobCopy();
        EXPECT_THROW(g->trimMemory(), Exception);
        EXPECT_EQ(input->getRawDataPtr<const void *>(), inputAddress);
        EXPECT_EQ(output->getRawDataPtr<const void *>(), outputAddress);
        EXPECT_EQ(g->getAllocationGeneration(), generation);
        EXPECT_EQ(g->getCaptureGeneration(), captureGeneration);
        EXPECT_EQ(runtime->getLiveAllocationCount(), liveAllocations);

        runtime->run(g);
        EXPECT_TRUE(output->equalData(data));
        EXPECT_NO_THROW(g->trimMemory());
        runtime->run(g);
        EXPECT_TRUE(output->equalData(data));
    }
    EXPECT_EQ(runtime->getLiveAllocationCount(), 0);
    EXPECT_EQ(runtime->getAllocationCount(), runtime->getDeallocationCount());
}

TEST(Graph, graph_locks_allocator_mode) {
    auto runtime = make_ref<TrackingCpuRuntimeObj>();

    Graph dynamic = make_ref<GraphObj>(runtime);
    dynamic->addTensor({4}, DataType::Float32)->setInput();
    dynamic->dataMalloc();
    EXPECT_THROW(dynamic->dataMalloc(true), Exception);
    EXPECT_THROW(dynamic->dataMalloc(false, 1024), Exception);

    Graph naive = make_ref<GraphObj>(runtime);
    naive->addTensor({4}, DataType::Float32)->setInput();
    naive->dataMalloc(true);
    EXPECT_THROW(naive->dataMalloc(), Exception);

    Graph fixed = make_ref<GraphObj>(runtime);
    fixed->addTensor({4}, DataType::Float32)->setInput();
    fixed->dataMalloc(false, 1024);
    EXPECT_NO_THROW(fixed->dataMalloc());
    EXPECT_THROW(fixed->dataMalloc(false, 2048), Exception);
    EXPECT_THROW(fixed->trimMemory(), Exception);
}

TEST(Graph, allocation_failure_does_not_lock_allocator_mode) {
    auto runtime = make_ref<TrackingCpuRuntimeObj>();

    Graph fixed = make_ref<GraphObj>(runtime);
    Tensor fixedInput = fixed->addTensor({4}, DataType::Float32);
    fixedInput->setInput();
    EXPECT_THROW(fixed->dataMalloc(false, 8), Exception);
    EXPECT_FALSE(fixedInput->hasData());
    EXPECT_NO_THROW(fixed->dataMalloc(false, 1024));
    EXPECT_TRUE(fixedInput->hasData());

    Graph allocationFailure = make_ref<GraphObj>(runtime);
    Tensor dynamicInput = allocationFailure->addTensor({4}, DataType::Float32);
    dynamicInput->setInput();
    runtime->failNextAlloc();
    EXPECT_THROW(allocationFailure->dataMalloc(false, 1024), std::bad_alloc);
    EXPECT_FALSE(dynamicInput->hasData());
    EXPECT_NO_THROW(allocationFailure->dataMalloc());
    EXPECT_TRUE(dynamicInput->hasData());
}

TEST(Graph, allocation_generation_tracks_blob_extent_changes) {
    auto runtime = make_ref<TrackingCpuRuntimeObj>();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor input = g->addTensor({1}, DataType::Float32);
    input->setInput();

    g->dataMalloc();
    const auto address = input->getRawDataPtr<const void *>();
    const auto generation = g->getAllocationGeneration();
    const auto allocationCount = runtime->getAllocationCount();

    input->setShape({2});
    g->dataMalloc();

    EXPECT_EQ(input->getRawDataPtr<const void *>(), address);
    EXPECT_EQ(runtime->getAllocationCount(), allocationCount);
    EXPECT_EQ(input->getDataBlob()->getBytes(), input->getBytes());
    EXPECT_GT(g->getAllocationGeneration(), generation);
}

TEST(Graph, fixed_pool_checks_capacity_and_heap_lifetime) {
    auto runtime = make_ref<TrackingCpuRuntimeObj>();
    {
        Graph tooSmall = make_ref<GraphObj>(runtime);
        tooSmall->addTensor({4}, DataType::Float32)->setInput();
        EXPECT_THROW(tooSmall->dataMalloc(false, 8), Exception);
    }

    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor input = g->addTensor({4}, DataType::Float32);
        input->setInput();
        g->dataMalloc(false, 1024);
        input->copyin(vector<float>{1, 2, 3, 4});

        const auto generation = g->getAllocationGeneration();
        Tensor oversized =
            make_ref<TensorObj>(Shape{300}, DataType::Float32, runtime);
        oversized->dataMalloc();
        EXPECT_THROW(g->cloneKV(oversized), Exception);
        EXPECT_EQ(g->getAllocationGeneration(), generation);

        runtime->failNextBlobCopy();
        EXPECT_THROW(g->cloneKV(input), Exception);
        EXPECT_EQ(g->getAllocationGeneration(), generation);

        auto clone = g->cloneKV(input);
        EXPECT_THROW(g->freeHeap(), Exception);
        clone.reset();
        EXPECT_NO_THROW(g->freeHeap());
    }
    EXPECT_EQ(runtime->getLiveAllocationCount(), 0);
    EXPECT_EQ(runtime->getAllocationCount(), runtime->getDeallocationCount());
}

TEST(Graph, owned_blobs_release_exactly_once) {
    auto runtime = make_ref<TrackingCpuRuntimeObj>();
    Tensor tensor =
        make_ref<TensorObj>(Shape{1, 2}, DataType::Float32, runtime);

    tensor->dataMalloc();
    EXPECT_EQ(runtime->getLiveAllocationCount(), 1);
    auto oldBlob = tensor->getDataBlob();

    tensor->setShape({8, 2});
    tensor->dataMalloc();
    EXPECT_EQ(runtime->getLiveAllocationCount(), 2);
    EXPECT_NE(tensor->getDataBlob(), oldBlob);

    oldBlob.reset();
    EXPECT_EQ(runtime->getLiveAllocationCount(), 1);

    tensor->setShape({2, 2});
    tensor->dataMalloc();
    EXPECT_EQ(runtime->getLiveAllocationCount(), 1);
    EXPECT_EQ(tensor->getDataBlob()->getBytes(), tensor->getBytes());

    {
        auto clone = tensor->clone(runtime);
        EXPECT_EQ(runtime->getLiveAllocationCount(), 2);
    }
    EXPECT_EQ(runtime->getLiveAllocationCount(), 1);

    {
        auto temporary = runtime->allocBlob(32);
        EXPECT_EQ(runtime->getLiveAllocationCount(), 2);
    }
    EXPECT_EQ(runtime->getLiveAllocationCount(), 1);

    tensor->freeData();
    EXPECT_EQ(runtime->getLiveAllocationCount(), 0);
    EXPECT_EQ(runtime->getAllocationCount(), runtime->getDeallocationCount());
}

TEST(Graph, blob_destructor_contains_deallocation_exceptions) {
    auto runtime = make_ref<TrackingCpuRuntimeObj>();
    auto blob = runtime->allocBlob(32);
    runtime->failNextDealloc();

    testing::internal::CaptureStderr();
    blob.reset();
    const auto error = testing::internal::GetCapturedStderr();

    EXPECT_NE(error.find("Error in ~BlobObj"), string::npos);
    EXPECT_EQ(runtime->getLiveAllocationCount(), 0);
    EXPECT_EQ(runtime->getAllocationCount(), runtime->getDeallocationCount());
}

TEST(Graph, lazy_allocator_views_keep_storage_alive) {
    auto runtime = make_ref<TrackingCpuRuntimeObj>();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor input = g->addTensor({1, 2}, DataType::Float32);
        Tensor weight = g->addTensor({2, 2}, DataType::Float32);
        input->setInput();
        weight->setWeight();
        auto matmul = g->addOp<MatmulObj>(input, weight, nullptr);
        matmul->getOutput()->setOutput();

        g->dataMalloc();
        EXPECT_EQ(runtime->getLiveAllocationCount(), 2);
        auto oldInputBlob = input->getDataBlob();

        input->setShape({8192, 2});
        g->shape_infer();
        g->dataMalloc();
        EXPECT_EQ(runtime->getLiveAllocationCount(), 3);

        oldInputBlob.reset();
        EXPECT_EQ(runtime->getLiveAllocationCount(), 2);
        g->validateMemory();
    }
    EXPECT_EQ(runtime->getLiveAllocationCount(), 0);
    EXPECT_EQ(runtime->getAllocationCount(), runtime->getDeallocationCount());
}

TEST(Graph, allocation_failure_leaves_tensor_without_data) {
    auto runtime = make_ref<TrackingCpuRuntimeObj>();
    Tensor tensor =
        make_ref<TensorObj>(Shape{1, 2}, DataType::Float32, runtime);
    tensor->dataMalloc();
    tensor->setShape({8192, 2});
    runtime->failNextAlloc();

    EXPECT_THROW(tensor->dataMalloc(), std::bad_alloc);
    EXPECT_FALSE(tensor->hasData());
    EXPECT_EQ(runtime->getLiveAllocationCount(), 0);

    tensor->dataMalloc();
    EXPECT_TRUE(tensor->hasData());
    EXPECT_EQ(runtime->getLiveAllocationCount(), 1);
    tensor->freeData();
    EXPECT_EQ(runtime->getLiveAllocationCount(), 0);
}

TEST(Graph, lazy_allocation_failure_leaves_activations_without_data) {
    auto runtime = make_ref<TrackingCpuRuntimeObj>();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor input = g->addTensor({1, 2}, DataType::Float32);
        Tensor weight = g->addTensor({2, 2}, DataType::Float32);
        input->setInput();
        weight->setWeight();
        auto matmul = g->addOp<MatmulObj>(input, weight, nullptr);
        Tensor output = matmul->getOutput();
        output->setOutput();

        g->dataMalloc();
        EXPECT_EQ(runtime->getLiveAllocationCount(), 2);
        input->setShape({8192, 2});
        g->shape_infer();
        runtime->failNextAlloc();

        EXPECT_THROW(g->dataMalloc(), std::bad_alloc);
        EXPECT_FALSE(input->hasData());
        EXPECT_FALSE(output->hasData());
        EXPECT_TRUE(weight->hasData());
        // The old high-watermark pool remains available for a retry.
        EXPECT_EQ(runtime->getLiveAllocationCount(), 2);
        EXPECT_THROW(runtime->run(g), Exception);

        g->dataMalloc();
        EXPECT_TRUE(input->hasData());
        EXPECT_TRUE(output->hasData());
        EXPECT_TRUE(weight->hasData());
        EXPECT_EQ(runtime->getLiveAllocationCount(), 2);
    }
    EXPECT_EQ(runtime->getLiveAllocationCount(), 0);
    EXPECT_EQ(runtime->getAllocationCount(), runtime->getDeallocationCount());
}

TEST(Graph, lazy_weight_allocation_failure_can_retry) {
    auto runtime = make_ref<TrackingCpuRuntimeObj>();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor input = g->addTensor({1, 2}, DataType::Float32);
        Tensor weight = g->addTensor({2, 2}, DataType::Float32);
        input->setInput();
        weight->setWeight();
        auto matmul = g->addOp<MatmulObj>(input, weight, nullptr);
        matmul->getOutput()->setOutput();

        runtime->failNextAlloc();
        EXPECT_THROW(g->dataMalloc(), std::bad_alloc);
        EXPECT_FALSE(input->hasData());
        EXPECT_FALSE(weight->hasData());
        EXPECT_FALSE(matmul->getOutput()->hasData());
        EXPECT_EQ(runtime->getLiveAllocationCount(), 0);

        g->dataMalloc();
        ASSERT_FALSE(runtime->getAllocationSizes().empty());
        EXPECT_EQ(runtime->getAllocationSizes().front(), weight->getBytes());
        EXPECT_TRUE(input->hasData());
        EXPECT_TRUE(weight->hasData());
        EXPECT_TRUE(matmul->getOutput()->hasData());
        EXPECT_EQ(runtime->getLiveAllocationCount(), 2);
    }
    EXPECT_EQ(runtime->getLiveAllocationCount(), 0);
    EXPECT_EQ(runtime->getAllocationCount(), runtime->getDeallocationCount());
}

TEST(Graph, lazy_weight_copy_failure_can_retry) {
    auto runtime = make_ref<TrackingCpuRuntimeObj>();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor input = g->addTensor({1, 2}, DataType::Float32);
        Tensor weight = g->addTensor({2, 2}, DataType::Float32);
        input->setInput();
        weight->setWeight();
        weight->dataMalloc();
        weight->copyin(vector<float>{1, 0, 0, 1});
        auto matmul = g->addOp<MatmulObj>(input, weight, nullptr);
        matmul->getOutput()->setOutput();

        runtime->clearAllocationSizes();
        runtime->failNextBlobCopy();
        EXPECT_THROW(g->dataMalloc(), Exception);
        ASSERT_EQ(runtime->getAllocationSizes().size(), 1);
        EXPECT_EQ(runtime->getAllocationSizes()[0], weight->getBytes());
        EXPECT_TRUE(weight->equalData(vector<float>{1, 0, 0, 1}));
        EXPECT_EQ(runtime->getLiveAllocationCount(), 1);

        g->dataMalloc();
        ASSERT_GE(runtime->getAllocationSizes().size(), 3);
        EXPECT_EQ(runtime->getAllocationSizes()[1], weight->getBytes());
        EXPECT_TRUE(weight->equalData(vector<float>{1, 0, 0, 1}));
        EXPECT_TRUE(input->hasData());
        EXPECT_TRUE(matmul->getOutput()->hasData());
        EXPECT_EQ(runtime->getLiveAllocationCount(), 2);
    }
    EXPECT_EQ(runtime->getLiveAllocationCount(), 0);
    EXPECT_EQ(runtime->getAllocationCount(), runtime->getDeallocationCount());
}

TEST(Graph, fixed_pool_rejects_dynamic_layout_changes) {
    auto runtime = make_ref<TrackingCpuRuntimeObj>();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor a = g->addTensor({4}, DataType::Float32);
        Tensor b = g->addTensor({4}, DataType::Float32);
        Tensor c = g->addTensor({4}, DataType::Float32);
        a->setInput();
        b->setInput();
        c->setInput();

        g->dataMalloc(false, 1024);
        a->copyin(vector<float>{1, 2, 3, 4});
        b->copyin(vector<float>{10, 20, 30, 40});
        c->copyin(vector<float>{100, 200, 300, 400});

        g->dataMalloc(false, 1024);
        EXPECT_TRUE(a->equalData(vector<float>{1, 2, 3, 4}));
        EXPECT_TRUE(b->equalData(vector<float>{10, 20, 30, 40}));
        EXPECT_TRUE(c->equalData(vector<float>{100, 200, 300, 400}));

        a->setShape({8});
        EXPECT_THROW(g->dataMalloc(false, 1024), Exception);
        EXPECT_TRUE(b->equalData(vector<float>{10, 20, 30, 40}));
        EXPECT_TRUE(c->equalData(vector<float>{100, 200, 300, 400}));

        a->setShape({4});
        g->dataMalloc(false, 1024);
        EXPECT_TRUE(a->equalData(vector<float>{1, 2, 3, 4}));
        EXPECT_TRUE(b->equalData(vector<float>{10, 20, 30, 40}));
        EXPECT_TRUE(c->equalData(vector<float>{100, 200, 300, 400}));
        EXPECT_EQ(runtime->getLiveAllocationCount(), 1);
    }
    EXPECT_EQ(runtime->getLiveAllocationCount(), 0);
    EXPECT_EQ(runtime->getAllocationCount(), runtime->getDeallocationCount());
}

TEST(Graph, runtime_rejects_unallocated_graph) {
    auto runtime = make_ref<TrackingCpuRuntimeObj>();
    Graph nullGraph;
    EXPECT_THROW(runtime->run(nullGraph), Exception);

    Graph g = make_ref<GraphObj>(runtime);
    Tensor input = g->addTensor({1, 2}, DataType::Float32);
    Tensor weight = g->addTensor({2, 2}, DataType::Float32);
    input->setInput();
    weight->setWeight();
    g->addOp<MatmulObj>(input, weight, nullptr);

    EXPECT_THROW(runtime->run(g), Exception);
    EXPECT_EQ(runtime->getLiveAllocationCount(), 0);
}

TEST(Graph, profiling_temporary_buffers_release_after_dynamic_shape) {
    auto runtime = make_ref<TrackingCpuRuntimeObj>();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor input = g->addTensor({1, 41}, DataType::Float32);
        Tensor weight = g->addTensor({41, 43}, DataType::Float32);
        input->setInput();
        weight->setWeight();
        auto matmul = g->addOp<MatmulObj>(input, weight, nullptr);
        Tensor output = matmul->getOutput();
        output->setOutput();
        g->dataMalloc(true);

        input->setShape({37, 41});
        g->shape_infer();
        EXPECT_GT(runtime->getPerfTime(g), 0);

        EXPECT_FALSE(input->hasData());
        EXPECT_FALSE(output->hasData());
        EXPECT_TRUE(weight->hasData());
        EXPECT_EQ(runtime->getLiveAllocationCount(), 1);
    }
    EXPECT_EQ(runtime->getLiveAllocationCount(), 0);
    EXPECT_EQ(runtime->getAllocationCount(), runtime->getDeallocationCount());
}

TEST(Graph, tensor_shape_allocation_safety) {
    auto runtime = make_ref<TrackingCpuRuntimeObj>();
    EXPECT_THROW(make_ref<TensorObj>(Shape{1, -1}, DataType::Float32, runtime),
                 Exception);
    EXPECT_THROW(make_ref<TensorObj>(Shape{INT32_MAX, INT32_MAX, INT32_MAX},
                                     DataType::Float32, runtime),
                 Exception);

    auto empty = make_ref<TensorObj>(Shape{0, 2}, DataType::Float32, runtime);
    empty->dataMalloc();
    EXPECT_EQ(empty->getBytes(), 0);
    EXPECT_EQ(empty->getDataBlob()->getBytes(), 0);
    EXPECT_NE(empty->getRawDataPtr<void *>(), nullptr);
    empty->freeData();
    EXPECT_EQ(runtime->getLiveAllocationCount(), 0);
}

TEST(Graph, topological) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor a = g->addTensor({1, 2, 3}, DataType::UInt32);
    Tensor b = g->addTensor({1, 2, 3}, DataType::UInt32);
    Tensor ab = g->addTensor({1, 2, 3}, DataType::UInt32);
    Tensor c = g->addTensor({1, 2, 3}, DataType::UInt32);
    Tensor abc = g->addTensor({1, 2, 3}, DataType::UInt32);
    Tensor d = g->addTensor({1, 2, 3}, DataType::UInt32);
    Tensor abcd = g->addTensor({1, 2, 3}, DataType::UInt32);
    Tensor e = g->addTensor({1, 2, 3}, DataType::UInt32);
    Tensor abcde = g->addTensor({1, 2, 3}, DataType::UInt32);

    auto ops = std::vector{
        g->addOpWithOutputs<AddObj>(abcd, e, abcde),
        g->addOpWithOutputs<AddObj>(abc, d, abcd),
        g->addOpWithOutputs<AddObj>(ab, c, abc),
        g->addOpWithOutputs<AddObj>(a, b, ab),
    };

    {
        auto p = ops.begin();
        auto q = g->getOperators().begin();
        while (p != ops.end()) {
            EXPECT_EQ(*p++, *q++);
        }
    }

    EXPECT_TRUE(g->topo_sort());

    {
        auto p = ops.rbegin();
        auto q = g->getOperators().begin();
        while (p != ops.rend()) {
            EXPECT_EQ(*p++, *q++);
        }
    }
} // namespace infini

TEST(Graph, perf_engine) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i0 = g->addTensor({1, 2, 3}, DataType::UInt32);
    Tensor w0 = g->addTensor({1, 3, 4}, DataType::UInt32);
    auto matmul = g->addOp<MatmulObj>(i0, w0, nullptr);

    g->dataMalloc();
    i0->copyin(vector<uint32_t>{1, 2, 3, 4, 5, 6});
    w0->copyin(vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    runtime->run(g, true, true);
    double perfTime = runtime->getPerfTime(g);
    // The example matmul takes 0.0036ms with one core
    EXPECT_GT(perfTime, 0);
    EXPECT_LT(perfTime, 0.01);
    // check answer
    auto ans = make_ref<TensorObj>(Shape{1, 2, 4}, DataType::UInt32, runtime);
    ans->dataMalloc();
    ans->copyin(vector<uint32_t>{38, 44, 50, 56, 83, 98, 113, 128});
    EXPECT_TRUE(matmul->getOutput()->equalData(ans));
}

TEST(Graph, test_tensor_id) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i0 = g->addTensor({1, 2, 3}, DataType::UInt32);
    Tensor w0 = g->addTensor({1, 3, 4}, DataType::UInt32);
    Tensor o0 = g->addTensor({1, 2, 4}, DataType::UInt32);
    g->dataMalloc();
    i0->copyin(vector<uint32_t>{1, 2, 3, 4, 5, 6});
    w0->copyin(vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto i1 = g->addTensor(i0->clone());
    auto matmul = g->addOpWithOutputs<MatmulObj>(i0, w0, o0);
    g->print();
    EXPECT_NE(i0->getGuid(), i1->getGuid());
    EXPECT_EQ(i0->getFuid(), i1->getFuid());
    EXPECT_NE(i0->getDataBlob(), nullptr);
    EXPECT_EQ(i1->getDataBlob(), nullptr);
}

TEST(Graph, test_OpVec_ctor) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i0 = g->addTensor({1, 2, 3}, DataType::UInt32);
    Tensor w0 = g->addTensor({1, 3, 4}, DataType::UInt32);
    Tensor o0 = g->addTensor({1, 2, 4}, DataType::UInt32);
    g->dataMalloc();
    i0->copyin(vector<uint32_t>{1, 2, 3, 4, 5, 6});
    w0->copyin(vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto o1 = g->addTensor(o0->clone());
    auto matmul = g->addOpWithOutputs<MatmulObj>(i0, w0, o0);
    g->addOp<ReluObj>(o1, nullptr);
    g->print();
    puts("=========");
    OpVec ops = g->getOperators();
    Graph g2 = make_ref<GraphObj>(runtime, ops);
    g2->print();
    // Check if the two tensors with the same FUID (o0,o1) remain only one in g2
    EXPECT_EQ(g2->getTensors().size(), 4u);
    EXPECT_EQ(g2->getOperators().size(), 2u);
    map<pair<int, int>, int> inputOutput2Cnt = {
        {{1, 0}, 2}, {{1, 1}, 1}, {{0, 1}, 1}};
    for (auto t : g2->getTensors()) {
        pair<int, int> key = {t->getTargets().size(),
                              t->getSource() != nullptr};
        EXPECT_GE(inputOutput2Cnt[key], 0);
        inputOutput2Cnt[key]--;
    }
    for (auto [u, v] : inputOutput2Cnt) {
        EXPECT_EQ(v, 0);
    }
}

} // namespace infini
