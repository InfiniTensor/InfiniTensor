#include "xpu/operator_timer.h"
#include "xpu/xpu_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/conv.h"
#include "operators/matmul.h"
#include "utils/data_generator.h"

namespace infini {
namespace opTimer {

double getPerfConvXPU(int n, int c, int h, int w, int f, int r, int s,
                       int padh, int padw, int strideh, int stridew,
                       int dilationh, int dilationw, int group,
                       const char *name) {
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);
    Runtime xpu = make_ref<XPURuntimeObj>();
    Graph gXpu = make_ref<GraphObj>(xpu);
    // Set input data on CPU in a CPU Graph
    IT_ASSERT(c % group == 0);
    Tensor i0Cpu = gCpu->addTensor({n, h, w, c}, DataType::Float32);
    Tensor w0Cpu = gCpu->addTensor({f, r, s, c / group}, DataType::Float32);
    // Malloc data for all tensors in a graph. Do we need implicit allocation?
    gCpu->dataMalloc();
    i0Cpu->setData(IncrementalGenerator());
    w0Cpu->setData(IncrementalGenerator());

    // Copy input tensors from CPU to Xpu
    Tensor i0XPU = gXpu->cloneTensor(i0Cpu);
    Tensor w0XPU = gXpu->cloneTensor(w0Cpu);
    // Build Xpu graph
    auto conv = gXpu->addOp<ConvObj>(i0XPU, w0XPU, nullptr, padh, padw,
                                      strideh, stridew, dilationh, dilationw);
    // allocate Xpu memory
    gXpu->dataMalloc();
    // Execute on Xpu
    bool tune = true;
    xpu->run(gXpu, tune);
    return xpu->getPerfTime(gXpu);
}

double getPerfMatmulXPU(int b, int m, int n, int k, const char *name) {
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);
    Runtime xpu = make_ref<XPURuntimeObj>();
    Graph gXpu = make_ref<GraphObj>(xpu);
    // Set input data on CPU in a CPU Graph
    Tensor i0Cpu = gCpu->addTensor({b, m, k}, DataType::Float32);
    Tensor w0Cpu = gCpu->addTensor({b, k, n}, DataType::Float32);
    // Malloc data for all tensors in a graph. Do we need implicit allocation?
    gCpu->dataMalloc();
    i0Cpu->setData(IncrementalGenerator());
    w0Cpu->setData(IncrementalGenerator());

    // Copy input tensors from CPU to Xpu
    Tensor i0XPU = gXpu->cloneTensor(i0Cpu);
    Tensor w0XPU = gXpu->cloneTensor(w0Cpu);
    // Build Xpu graph
    auto conv = gXpu->addOp<MatmulObj>(i0XPU, w0XPU, nullptr);
    // allocate Xpu memory
    gXpu->dataMalloc();
    // Execute on Xpu
    bool tune = true;
    xpu->run(gXpu, tune);
    return xpu->getPerfTime(gXpu);
}

} // namespace opTimer
} // namespace infini
