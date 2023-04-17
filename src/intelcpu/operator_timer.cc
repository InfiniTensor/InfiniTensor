#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "intelcpu/mkl_runtime.h"
#include "operators/conv.h"
#include "operators/matmul.h"
#include "utils/data_generator.h"

namespace infini {
namespace opTimer {

double getPerfConvMkl(int n, int c, int h, int w, int f, int r, int s, int padh,
                      int padw, int strideh, int stridew, int dilationh,
                      int dilationw, int group) {
    // const auto &[n, c, h, w, f, r, s, padh, padw, strideh, stridew,
    // dilationh, dilationw, group] =
    //     tuple{1, 512, 14, 14, 512, 3, 3, 2, 2, 1, 1, 2, 2, 1};
    Runtime runtime = MklRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph g = make_ref<GraphObj>(runtime);
    IT_ASSERT(c % group == 0);
    Tensor i0 = g->addTensor({n, c, h, w}, DataType::Float32);
    Tensor w0 = g->addTensor({f, c / group, r, s}, DataType::Float32);
    auto conv = g->addOp<ConvObj>(i0, w0, nullptr, padh, padw, strideh, stridew,
                                  dilationh, dilationw);
    // Malloc data for all tensors in a graph. Do we need implicit allocation?
    g->dataMalloc();
    i0->setData(IncrementalGenerator());
    w0->setData(IncrementalGenerator());

    bool tune = true;
    runtime->run(g, tune);
    return runtime->getPerfTime(g);
}

double getPerfConvTransposed2dMkl(int n, int c, int h, int w, int f, int r,
                                  int s, int padh, int padw, int strideh,
                                  int stridew, int dilationh, int dilationw,
                                  int oph, int opw, int group) {
    // const auto &[n, c, h, w, f, r, s, padh, padw, strideh, stridew,
    // dilationh, dilationw, group] =
    //     tuple{1, 512, 14, 14, 512, 3, 3, 2, 2, 1, 1, 2, 2, 1};
    Runtime runtime = MklRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph g = make_ref<GraphObj>(runtime);
    // Set input data on CPU in a CPU Graph
    IT_ASSERT(c % group == 0);
    Tensor i0 = g->addTensor({n, f, h, w}, DataType::Float32);
    Tensor w0 = g->addTensor({f, c / group, r, s}, DataType::Float32);
    auto conv = g->addOp<ConvTransposed2dObj>(i0, w0, nullptr, padh, padw,
                                              strideh, stridew, dilationh,
                                              dilationw, oph, opw, group);
    // Malloc data for all tensors in a graph. Do we need implicit allocation?
    g->dataMalloc();
    i0->setData(IncrementalGenerator());
    w0->setData(IncrementalGenerator());

    bool tune = true;
    runtime->run(g, tune);
    return runtime->getPerfTime(g);
}

double getPerfMatmulMkl(int b, int m, int n, int k) {
    // const auto &[n, c, h, w, f, r, s, padh, padw, strideh, stridew,
    // dilationh, dilationw, group] =
    //     tuple{1, 512, 14, 14, 512, 3, 3, 2, 2, 1, 1, 2, 2, 1};
    Runtime runtime = MklRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph g = make_ref<GraphObj>(runtime);
    // Set input data on CPU in a CPU Graph
    Tensor i0 = g->addTensor({b, m, k}, DataType::Float32);
    Tensor w0 = g->addTensor({b, k, n}, DataType::Float32);
    auto conv = g->addOp<MatmulObj>(i0, w0, nullptr);
    // Malloc data for all tensors in a graph. Do we need implicit allocation?
    g->dataMalloc();
    i0->setData(IncrementalGenerator());
    w0->setData(IncrementalGenerator());

    bool tune = true;
    runtime->run(g, tune);
    return runtime->getPerfTime(g);
}

} // namespace opTimer
} // namespace infini
