#include "bang/bang_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

template <class T> void testArange() {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto bangRuntime = make_ref<BangRuntimeObj>();

    float start = 0.0;
    float step = 2.0;
    int length = 10;
    // GPU
    Graph bangGraph = make_ref<GraphObj>(bangRuntime);
    auto gpuOp = bangGraph->addOp<T>(start, step, length, nullptr);
    bangGraph->dataMalloc();
    bangRuntime->run(bangGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    outputGpu2Cpu->printData();
    EXPECT_TRUE(1);
}

TEST(cnnl_Arange, run) { testArange<ArangeObj>(); }

} // namespace infini
