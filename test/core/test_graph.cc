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
        EXPECT_EQ(runtime->getLiveAllocationCount(), 1);
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
        EXPECT_EQ(runtime->getLiveAllocationCount(), 2);

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
