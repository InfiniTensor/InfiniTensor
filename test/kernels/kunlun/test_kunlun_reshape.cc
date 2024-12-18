#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "kunlun/kunlun_runtime.h"
#include "operators/reshape.h"

#include "test.h"

namespace infini {

TEST(KUNLUN_Reshape, run) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto kunlunRuntime = make_ref<KUNLUNRuntimeObj>();

    // Build input data on CPU
    Tensor icpu =
        make_ref<TensorObj>(Shape{2, 3, 3, 4}, DataType::Float32, cpuRuntime);
    icpu->dataMalloc();
    icpu->setData(IncrementalGenerator());

    // Build KUNLUN graph
    Graph g = make_ref<GraphObj>(kunlunRuntime);
    auto i = g->cloneTensor(icpu);
    auto op = g->addOp<ReshapeObj>(i, nullptr, Shape{3, 2, 4, 3});

    // allocate KUNLUN memory
    g->dataMalloc();
    i->setData(IncrementalGenerator());

    // Execute on KUNLUN
    kunlunRuntime->run(g);

    // clone KUNLUN output to CPU
    auto o = op->getOutput();
    auto ocpu = o->clone(cpuRuntime);

    //  check results on CPU
    EXPECT_TRUE(ocpu->equalData(icpu));
}

TEST(KUNLUN_Flatten, run) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto kunlunRuntime = make_ref<KUNLUNRuntimeObj>();

    // Build input data on CPU
    Tensor icpu =
        make_ref<TensorObj>(Shape{2, 3, 3, 4}, DataType::Float32, cpuRuntime);
    icpu->dataMalloc();
    icpu->setData(IncrementalGenerator());

    // Build KUNLUN graph
    Graph g = make_ref<GraphObj>(kunlunRuntime);
    auto i = g->cloneTensor(icpu);
    auto op = g->addOp<FlattenObj>(i, nullptr, 2);

    // allocate KUNLUN memory
    g->dataMalloc();
    i->setData(IncrementalGenerator());

    // Execute on KUNLUN
    kunlunRuntime->run(g);

    // clone KUNLUN output to CPU
    auto o = op->getOutput();
    auto ocpu = o->clone(cpuRuntime);

    //  check results on CPU
    EXPECT_TRUE(ocpu->equalData(icpu));
}

TEST(KUNLUN_Identity, run) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto kunlunRuntime = make_ref<KUNLUNRuntimeObj>();

    // Build input data on CPU
    Tensor icpu =
        make_ref<TensorObj>(Shape{2, 3, 3, 4}, DataType::Float32, cpuRuntime);
    icpu->dataMalloc();
    icpu->setData(IncrementalGenerator());

    // Build KUNLUN graph
    Graph g = make_ref<GraphObj>(kunlunRuntime);
    auto i = g->cloneTensor(icpu);
    auto op = g->addOp<IdentityObj>(i, nullptr);

    // allocate KUNLUN memory
    g->dataMalloc();
    i->setData(IncrementalGenerator());

    // Execute on KUNLUN
    kunlunRuntime->run(g);

    // clone KUNLUN output to CPU
    auto o = op->getOutput();
    auto ocpu = o->clone(cpuRuntime);

    //  check results on CPU
    EXPECT_TRUE(ocpu->equalData(icpu));
}
} // namespace infini