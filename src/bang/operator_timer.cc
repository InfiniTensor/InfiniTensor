#include "bang/operator_timer.h"
#include "bang/bang_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/conv.h"
#include "operators/matmul.h"
#include "utils/data_generator.h"

namespace infini {
namespace opTimer {

double getPerfConvCnnl(int n, int c, int h, int w, int f, int r, int s,
                       int padh, int padw, int strideh, int stridew,
                       int dilationh, int dilationw, int group,
                       const char *name) {
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);
    Runtime bang = make_ref<BangRuntimeObj>();
    Graph gBang = make_ref<GraphObj>(bang);
    // Set input data on CPU in a CPU Graph
    IT_ASSERT(c % group == 0);
    Tensor i0Cpu = gCpu->addTensor({n, h, w, c}, DataType::Float32);
    Tensor w0Cpu = gCpu->addTensor({f, r, s, c / group}, DataType::Float32);
    // Malloc data for all tensors in a graph. Do we need implicit allocation?
    gCpu->dataMalloc();
    i0Cpu->setData(IncrementalGenerator());
    w0Cpu->setData(IncrementalGenerator());

    // Copy input tensors from CPU to Bang
    Tensor i0Bang = gBang->cloneTensor(i0Cpu);
    Tensor w0Bang = gBang->cloneTensor(w0Cpu);
    // Build Bang graph
    auto conv = gBang->addOp<ConvObj>(i0Bang, w0Bang, nullptr, padh, padw,
                                      strideh, stridew, dilationh, dilationw);
    // allocate Bang memory
    gBang->dataMalloc();
    // Execute on Bang
    bool tune = true;
    bang->run(gBang, tune);
    return bang->getPerfTime(gBang);
}

double getPerfMatmulCnnl(int b, int m, int n, int k, const char *name) {
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);
    Runtime bang = make_ref<BangRuntimeObj>();
    Graph gBang = make_ref<GraphObj>(bang);
    // Set input data on CPU in a CPU Graph
    Tensor i0Cpu = gCpu->addTensor({b, m, k}, DataType::Float32);
    Tensor w0Cpu = gCpu->addTensor({b, k, n}, DataType::Float32);
    // Malloc data for all tensors in a graph. Do we need implicit allocation?
    gCpu->dataMalloc();
    i0Cpu->setData(IncrementalGenerator());
    w0Cpu->setData(IncrementalGenerator());

    // Copy input tensors from CPU to Bang
    Tensor i0Bang = gBang->cloneTensor(i0Cpu);
    Tensor w0Bang = gBang->cloneTensor(w0Cpu);
    // Build Bang graph
    auto conv = gBang->addOp<MatmulObj>(i0Bang, w0Bang, nullptr);
    // allocate Bang memory
    gBang->dataMalloc();
    // Execute on Bang
    bool tune = true;
    bang->run(gBang, tune);
    return bang->getPerfTime(gBang);
}

} // namespace opTimer
} // namespace infini
