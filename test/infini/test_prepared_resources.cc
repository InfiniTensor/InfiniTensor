#include "../../src/kernels/infiniops/lowering.h"
#include "core/infini_runtime.h"
#include "operators/infini_op.h"
#include "operators/pooling.h"
#include "operators/unary.h"
#include "test.h"
#include <cstring>

namespace infini::scratch_test {
struct Key;
using namespace infiniops;
std::vector<std::weak_ptr<infiniops::Workspace>> slots;
std::vector<std::weak_ptr<void>> descriptors;
int makes = 0, calls = 0;
bool unsafe = false, failCapture = false, failAfterSubmit = false;
void copy(const Handle &handle, void *dst, const void *src, size_t bytes,
          ::infini::rt::Device::Type device) {
    if (device == ::infini::rt::Device::Type::kCpu)
        std::memcpy(dst, src, bytes);
    else
        IT_ASSERT(::infini::rt::runtime::MemcpyAsync(
                      dst, src, bytes,
                      ::infini::rt::runtime::kMemcpyDeviceToDevice,
                      reinterpret_cast<::infini::rt::runtime::Stream>(
                          handle.stream())) == ::infini::rt::runtime::kSuccess);
}
} // namespace infini::scratch_test

// A copy provider with a real, nonzero workspace exercises the same typed
// preparation ABI as generated SDK bindings. It is confined to this test.
namespace infini::ops {
template <> class Operator<::infini::scratch_test::Key> : public OperatorBase {
    std::shared_ptr<void> marker = std::make_shared<int>(0);

  public:
    Operator() {
        scratch_test::descriptors.push_back(marker);
        ++scratch_test::makes;
    }
    size_t workspace_size_in_bytes() const override { return 64; }
    bool capture_safe() const override { return !scratch_test::unsafe; }
    void operator()(const Handle &handle, const Tensor input,
                    Tensor out) const {
        ++scratch_test::calls;
        IT_ASSERT(!scratch_test::failCapture || scratch_test::calls != 2,
                  "Injected capture failure");
        IT_ASSERT(handle.workspace() && handle.workspace_size_in_bytes() >= 64);
        scratch_test::copy(handle, handle.workspace(), input.data(), 64,
                           input.device().type());
        scratch_test::copy(handle, out.data(), handle.workspace(), 64,
                           input.device().type());
        IT_ASSERT(!scratch_test::failAfterSubmit,
                  "Injected error after submission");
    }
};
} // namespace infini::ops

namespace infini::scratch_test {
struct Key {
    static std::unique_ptr<::infini::ops::Operator<Key>>
    Make(const ::infini::ops::Config &, const TensorView &,
         const TensorView &) {
        return std::make_unique<::infini::ops::Operator<Key>>();
    }
};
} // namespace infini::scratch_test

namespace infini {
namespace {
using infiniops::getArgument;
using infiniops::Handle;
using infiniops::prepareGraph;
using infiniops::prepareTyped;
using infiniops::Registry;
using infiniops::Step;
using infiniops::TensorView;
using infiniops::Values;
using infiniops::WorkspaceAllocator;
using Bindings = std::map<std::string, TensorVec>;
Ref<InfiniRuntimeObj> runtime(size_t capacity = 2) {
    return make_ref<InfiniRuntimeObj>(
        string(::infini::rt::Device::StringFromType(
            ::infini::rt::runtime_device_type())),
        0, capacity);
}
[[maybe_unused]] const bool registered = [] {
    Registry::instance().add(
        "PreparedScratchProbe", {{"input", "Tensor"}, {"out", "Tensor"}},
        [](const Values &values, const WorkspaceAllocator &allocate) {
            auto track = [&](size_t bytes) {
                auto slot = allocate(bytes);
                scratch_test::slots.push_back(slot);
                return slot;
            };
            return prepareTyped<scratch_test::Key>(
                track, getArgument<TensorView>(values[0]),
                getArgument<TensorView>(values[1]));
        });
#if !INFINITENSOR_TEST_HAS_ATEN
    // The CPU-only SDK lacks MaxPool. A 1x1 copy provider supplies only the
    // arithmetic for this fixture; graph/lowering/planning remain production.
    Registry::instance().add(
        "MaxPool2dWithIndices",
        {{"input", "Tensor"},
         {"kernel", "std::vector<int64_t>"},
         {"stride", "std::vector<int64_t>"},
         {"padding", "std::vector<int64_t>"},
         {"dilation", "std::vector<int64_t>"},
         {"ceil", "bool"},
         {"out", "Tensor"},
         {"indices", "Tensor"}},
        [](const Values &v, const WorkspaceAllocator &) -> Step {
            IT_ASSERT(std::get<std::vector<int64_t>>(v[1]) ==
                      std::vector<int64_t>({1, 1}));
            auto in = std::get<TensorView>(v[0]),
                 out = std::get<TensorView>(v[6]);
            auto indices = std::get<TensorView>(v[7]);
            IT_ASSERT(in.device().type() == ::infini::rt::Device::Type::kCpu);
            return {[in, out, indices](const Handle &) mutable {
                        std::memcpy(out.data(), in.data(),
                                    in.numel() * sizeof(float));
                        auto *p = static_cast<int64_t *>(indices.data());
                        for (size_t i = 0; i < indices.numel(); ++i)
                            p[i] = i;
                    },
                    false, "CPU test-only 1x1 MaxPool"};
        });
#endif
    return true;
}();

void resetProbes() {
    scratch_test::slots.clear();
    scratch_test::descriptors.clear();
    scratch_test::makes = scratch_test::calls = 0;
    scratch_test::unsafe = scratch_test::failCapture =
        scratch_test::failAfterSubmit = false;
}
Graph probe(const Ref<InfiniRuntimeObj> &rt, int count) {
    auto g = make_ref<GraphObj>(rt);
    auto x = g->addTensor({16}, DataType::Float32);
    x->setInput();
    auto y = x;
    for (int i = 0; i < count; ++i) {
        auto out = g->addTensor({16}, DataType::Float32);
        g->addOp<InfiniOpObj>("PreparedScratchProbe", Bindings{{"input", {y}}},
                              Bindings{{"out", {out}}});
        y = out;
    }
    y->setOutput();
    g->dataMalloc();
    x->copyin(vector<float>(16, 3));
    return g;
}
void expectReleased() {
    for (const auto &p : scratch_test::slots)
        EXPECT_TRUE(p.expired());
    for (const auto &p : scratch_test::descriptors)
        EXPECT_TRUE(p.expired());
}
TEST(PreparedResourcesTest, LayoutCapacityDoesNotDependOnArgumentOrder) {
    auto rt = runtime();
    for (bool reverse : {false, true}) {
        infiniops::Builder layout(
            [](size_t) -> std::shared_ptr<void> {
                throw std::runtime_error(
                    "Layout must not allocate device memory");
            },
            0, true);
        layout.beginNode();
        layout.temporary(reverse ? 256 : 8);
        layout.temporary(reverse ? 8 : 256);
        ASSERT_EQ(layout.temporaryPeak(), 512u);
        infiniops::Builder real(
            [&](size_t bytes) { return rt->acquireWorkspace(bytes); },
            layout.temporaryPeak());
        real.beginNode();
        EXPECT_NO_THROW(real.temporary(reverse ? 8 : 256));
        EXPECT_NO_THROW(real.temporary(reverse ? 256 : 8));
    }
}
TEST(PreparedResourcesTest, WorkspacePeakIsIndependentOfNodeCountAndCached) {
    for (int count : {1, 8, 32}) {
        resetProbes();
        auto rt = runtime();
        auto g = probe(rt, count);
        rt->run(g);
        ASSERT_EQ(scratch_test::makes, count); // layout pass must not call Make
        ASSERT_EQ(scratch_test::slots.size(), size_t(count));
        auto first = scratch_test::slots[0].lock();
        ASSERT_TRUE(first && first->data);
        EXPECT_EQ(first->bytes, 64u);
        std::cout << nlohmann::json{{"case", "workspace"},
                                    {"nodes", count},
                                    {"workspace_bytes", first->bytes},
                                    {"prepared_instances", scratch_test::makes}}
                         .dump()
                  << std::endl;
        for (const auto &p : scratch_test::slots)
            EXPECT_EQ(p.lock(), first);
        first.reset();
        rt->run(g);
        EXPECT_EQ(scratch_test::makes, count);
        EXPECT_EQ(g->getOutputs()[0]->copyout<float>(), vector<float>(16, 3));
        rt->clearGraphCache();
        expectReleased();
    }
}
TEST(PreparedResourcesTest, RealMaxPoolTemporaryPeakDoesNotAccumulate) {
    for (int count : {1, 8, 32}) {
        auto rt = runtime();
        auto g = make_ref<GraphObj>(rt);
        auto x = g->addTensor({1, 1, 256, 256}, DataType::Float32);
        x->setInput();
        auto y = x;
        for (int i = 0; i < count; ++i)
            y = g->addOp<MaxPoolObj>(y, nullptr, 1, 1, 1, 1, 0, 0, 1, 1, 0)
                    ->getOutput();
        y->setOutput();
        g->dataMalloc();
        x->copyin(vector<float>(256 * 256, 3));
        vector<size_t> requests;
        vector<std::weak_ptr<void>> allocations;
        auto plan = prepareGraph(g, rt.get(), [&](size_t bytes) {
            requests.push_back(bytes);
            auto p = rt->acquireWorkspace(bytes);
            allocations.push_back(p);
            return p;
        });
        EXPECT_EQ(plan->plan.temporaryBytes, 256u * 256 * sizeof(int64_t));
        EXPECT_EQ(requests, vector<size_t>{256u * 256 * sizeof(int64_t)});
        std::cout << nlohmann::json{{"case", "maxpool"},
                                    {"nodes", count},
                                    {"helper_bytes", plan->plan.temporaryBytes},
                                    {"workspace_bytes",
                                     plan->plan.workspace->bytes},
                                    {"allocation_count", requests.size()}}
                         .dump()
                  << std::endl;
        Handle handle;
        handle.set_stream(rt->getStream());
        plan->plan.run(handle);
        rt->sync();
        EXPECT_EQ(y->copyout<float>(), vector<float>(256 * 256, 3));
        plan.reset();
        for (const auto &p : allocations)
            EXPECT_TRUE(p.expired());
        rt->run(g);
        rt->run(g);
        EXPECT_EQ(y->copyout<float>(), vector<float>(256 * 256, 3));
        rt->clearGraphCache();
    }
}
TEST(PreparedResourcesTest, AllocationFailureReleasesPreparedDescriptors) {
    resetProbes();
    auto rt = runtime();
    auto g = probe(rt, 8);
    EXPECT_THROW(prepareGraph(g, rt.get(),
                              [](size_t) -> std::shared_ptr<void> {
                                  throw std::bad_alloc();
                              }),
                 std::bad_alloc);
    EXPECT_EQ(scratch_test::makes, 8);
    expectReleased();
    EXPECT_NO_THROW(rt->run(g));
    rt->clearGraphCache();
    expectReleased();
}
TEST(PreparedResourcesTest, LaterFailureReleasesAlreadyAllocatedHelperStorage) {
    resetProbes();
    auto rt = runtime();
    auto g = make_ref<GraphObj>(rt);
    auto x = g->addTensor({1, 1, 256, 256}, DataType::Float32);
    x->setInput();
    auto pooled = g->addOp<MaxPoolObj>(x, nullptr, 1, 1, 1, 1, 0, 0, 1, 1, 0)
                      ->getOutput();
    auto y = g->addTensor({16}, DataType::Float32);
    y->setOutput();
    g->addOp<InfiniOpObj>("PreparedScratchProbe", Bindings{{"input", {pooled}}},
                          Bindings{{"out", {y}}});
    g->dataMalloc();
    int requests = 0;
    std::weak_ptr<void> helper;
    EXPECT_THROW(prepareGraph(g, rt.get(),
                              [&](size_t bytes) {
                                  if (++requests == 2)
                                      throw std::bad_alloc();
                                  auto data = rt->acquireWorkspace(bytes);
                                  helper = data;
                                  return data;
                              }),
                 std::bad_alloc);
    EXPECT_EQ(requests, 2);
    EXPECT_TRUE(helper.expired());
    expectReleased();
}
TEST(PreparedResourcesTest, ClearSynchronizesAfterFailedOrdinarySubmission) {
    resetProbes();
    auto rt = runtime();
    auto g = probe(rt, 1);
    scratch_test::failAfterSubmit = true;
    EXPECT_THROW(rt->run(g), std::exception);
    rt->clearGraphCache();
    expectReleased();
    scratch_test::failAfterSubmit = false;
    EXPECT_NO_THROW(rt->run(g));
}
TEST(PreparedResourcesTest, FailedSubmissionCanSwitchGraphsWithoutExplicitClear) {
    resetProbes();
    auto rt = runtime();
    auto first = probe(rt, 1), second = probe(rt, 1);
    scratch_test::failAfterSubmit = true;
    EXPECT_THROW(rt->run(first), std::exception);
    scratch_test::failAfterSubmit = false;
    EXPECT_NO_THROW(rt->run(second));
    ASSERT_EQ(scratch_test::slots.size(), 2u);
    EXPECT_TRUE(scratch_test::slots.front().expired());
    EXPECT_EQ(second->getOutputs()[0]->copyout<float>(), vector<float>(16, 3));
    rt->clearGraphCache();
    expectReleased();
}
TEST(PreparedResourcesTest, FailedSubmissionAllowsImmediateGraphDestruction) {
    resetProbes();
    auto rt = runtime();
    auto first = probe(rt, 1);
    scratch_test::failAfterSubmit = true;
    EXPECT_THROW(rt->run(first), std::exception);
    first.reset();
    expectReleased();
    scratch_test::failAfterSubmit = false;
    auto second = probe(rt, 1);
    EXPECT_NO_THROW(rt->run(second));
    EXPECT_EQ(second->getOutputs()[0]->copyout<float>(), vector<float>(16, 3));
    rt->clearGraphCache();
    expectReleased();
}
#if INFINITENSOR_INFINIRT_HAS_GRAPH_API
TEST(PreparedResourcesTest, ExplicitClearAndLruReleasePreparedAllocations) {
    resetProbes();
    auto rt = runtime(1);
    auto first = probe(rt, 1), second = probe(rt, 1);
    rt->runWithGraph(first);
    rt->runWithGraph(second);
    ASSERT_EQ(scratch_test::slots.size(), 2u);
    EXPECT_TRUE(scratch_test::slots[0].expired());
    EXPECT_FALSE(scratch_test::slots[1].expired());
    rt->clearGraphCache();
    EXPECT_EQ(rt->getGraphCacheSize(), 0u);
    expectReleased();
    EXPECT_NO_THROW(rt->run(first));
}
TEST(PreparedResourcesTest,
     RejectionAndCaptureFailureReleasePreparedAllocations) {
    for (bool reject : {false, true}) {
        resetProbes();
        auto rt = runtime();
        auto g = probe(rt, 1);
        scratch_test::unsafe = reject;
        scratch_test::failCapture = !reject;
        EXPECT_THROW(rt->runWithGraph(g), std::exception);
        EXPECT_EQ(rt->getGraphCacheSize(), 0u);
        expectReleased();
        scratch_test::unsafe = scratch_test::failCapture = false;
        EXPECT_NO_THROW(rt->runWithGraph(g));
        rt->clearGraphCache();
        expectReleased();
    }
}
#endif
#if INFINITENSOR_TEST_HAS_ATEN
TEST(PreparedResourcesTest, HardSigmoidConstantsSurviveScratchReuse) {
    auto rt = runtime();
    auto g = make_ref<GraphObj>(rt);
    auto x = g->addTensor({1, 16}, DataType::Float32);
    x->setInput();
    vector<float> expected;
    for (int i = 0; i < 16; ++i)
        expected.push_back(i / 16.f);
    auto y = x;
    for (int i = 0; i < 8; ++i) {
        const float alpha = .8f + .01f * i, beta = .01f * i;
        y = g->addOp<HardSigmoidObj>(y, nullptr, alpha, beta)->getOutput();
        for (auto &value : expected)
            value = std::clamp(alpha * value + beta, 0.f, 1.f);
    }
    y->setOutput();
    g->dataMalloc();
    vector<float> input;
    for (int i = 0; i < 16; ++i)
        input.push_back(i / 16.f);
    x->copyin(input);
    rt->run(g);
    EXPECT_TRUE(
        y->clone(NativeCpuRuntimeObj::getInstance())->equalData(expected));
    rt->run(g);
    EXPECT_TRUE(
        y->clone(NativeCpuRuntimeObj::getInstance())->equalData(expected));
}
#endif
} // namespace
} // namespace infini
