#include "kunlun/operator_timer.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/conv.h"
#include "operators/matmul.h"
#include "utils/data_generator.h"
#include "kunlun/kunlun_runtime.h"

namespace infini {
namespace opTimer {

double getPerfConvKunlun(int n, int c, int h, int w, int f, int r, int s, int padh,
                      int padw, int strideh, int stridew, int dilationh,
                      int dilationw, int group, const char *name) {
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);
    Runtime kunlun = make_ref<KUNLUNRuntimeObj>();
    Graph gKunlun = make_ref<GraphObj>(kunlun);
    // Set input data on CPU in a CPU Graph
    IT_ASSERT(c % group == 0);
    Tensor i0Cpu = gCpu->addTensor({n, h, w, c}, DataType::Float32);
    Tensor w0Cpu = gCpu->addTensor({f, r, s, c / group}, DataType::Float32);
    // Malloc data for all tensors in a graph. Do we need implicit allocation?
    gCpu->dataMalloc();
    i0Cpu->setData(IncrementalGenerator());
    w0Cpu->setData(IncrementalGenerator());

    // Copy input tensors from CPU to Kunlun
    Tensor i0Kunlun = gKunlun->cloneTensor(i0Cpu);
    Tensor w0Kunlun = gKunlun->cloneTensor(w0Cpu);
    // Build Kunlun graph
    auto conv = gKunlun->addOp<ConvObj>(i0Kunlun, w0Kunlun, nullptr, padh, padw, strideh,
                                     stridew, dilationh, dilationw);
    // allocate Kunlun memory
    gKunlun->dataMalloc();
    // Execute on Kunlun
    bool tune = true;
    kunlun->run(gKunlun, tune);
    return kunlun->getPerfTime(gKunlun);
}

double getPerfMatmulKunlun(int b, int m, int n, int k, const char *name) {
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);
    Runtime kunlun = make_ref<KUNLUNRuntimeObj>();
    Graph gKunlun = make_ref<GraphObj>(kunlun);
    // Set input data on CPU in a CPU Graph
    Tensor i0Cpu = gCpu->addTensor({b, m, k}, DataType::Float32);
    Tensor w0Cpu = gCpu->addTensor({b, k, n}, DataType::Float32);
    // Malloc data for all tensors in a graph. Do we need implicit allocation?
    gCpu->dataMalloc();
    i0Cpu->setData(IncrementalGenerator());
    w0Cpu->setData(IncrementalGenerator());

    // Copy input tensors from CPU to Kunlun
    Tensor i0Kunlun = gKunlun->cloneTensor(i0Cpu);
    Tensor w0Kunlun = gKunlun->cloneTensor(w0Cpu);
    // Build Kunlun graph
    auto conv = gKunlun->addOp<MatmulObj>(i0Kunlun, w0Kunlun, nullptr);
    // allocate Kunlun memory
    gKunlun->dataMalloc();
    // Execute on Kunlun
    bool tune = true;
    kunlun->run(gKunlun, tune);
    return kunlun->getPerfTime(gKunlun);
}

} // namespace opTimer
} // namespace infini
