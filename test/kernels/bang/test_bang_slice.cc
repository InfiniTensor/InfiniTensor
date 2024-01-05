#include "bang/bang_runtime.h"
#include "core/graph.h"
#include "core/runtime.h"
#include "operators/slice.h"
#include "test.h"

namespace infini {
TEST(BANG_Slice, run) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto bangRuntime = make_ref<BangRuntimeObj>();

    // Build input data on CPU
    Tensor icpu =
        make_ref<TensorObj>(Shape{3, 2, 1, 5}, DataType::Float32, cpuRuntime);
    icpu->dataMalloc();
    icpu->setData(IncrementalGenerator());

    // Build CUDA graph;
    Graph g = make_ref<GraphObj>(bangRuntime);
    auto i = g->cloneTensor(icpu);
    auto op =
        g->addOp<SliceObj>(i, nullptr, vector<int>{1, 1}, vector<int>{2, 5},
                           vector<int>{0, 3}, std::nullopt);

    // allocate CUDA memory
    g->dataMalloc();
    i->setData(IncrementalGenerator());

    // Execute on CUDA
    bangRuntime->run(g);

    // clone CUDA output to CPU
    auto o = op->getOutput();
    auto cpuo = o->clone(cpuRuntime);
    // bangPrintTensor(o);
    //  check results on CPU
    EXPECT_TRUE(cpuo->equalData(vector<float>{11, 12, 13, 14, 16, 17, 18, 19}));
}
} // namespace infini
