#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/conv.h"
#include "operators/matmul.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {
TEST(Case, fsrcnn) {
    auto cuda = make_ref<CudaRuntimeObj>();
    Graph g = make_ref<GraphObj>(cuda);
    auto input = g->addTensor({1, 1, 32, 32});
    vector<tuple<string, int, int, bool>> configs = {
        {"Conv", 56, 5, true},  {"Conv", 12, 1, true},
        {"Conv", 12, 3, false}, {"Conv", 12, 3, false},
        {"Conv", 12, 3, false}, {"Conv", 12, 3, true},
        {"Conv", 56, 1, true},  {"ConvTranposed", 56, 9, false}};
    auto x = input;
    for (auto &[op, f, r, pRelu] : configs) {
        if (op == "Conv") {
            auto w = g->addTensor({f, x->getDims()[1], r, r});
            x = g->addOp<ConvObj>(x, w, nullptr, r / 2, r / 2)->getOutput();
            if (pRelu) {
                x = g->addOp<ReluObj>(x, nullptr)->getOutput();
            }
        } else if (op == "ConvTranposed") {
            IT_ASSERT(r == 9);
            auto w = g->addTensor({x->getDims()[1], f, r, r});
            x = g->addOp<ConvTransposed2dObj>(x, w, nullptr, 3, 3, 4, 4, 1, 1,
                                              1, 1)
                    ->getOutput();
        }
    }
    g->print();
    g->dataMalloc();
    cuda->run(g, true);
    cuda->getPerfTime(g, true);
    cudaProfilerStart();
    printf("E2E time %.3lf\n",
           timeit([&]() { cuda->runWithoutSync(g); }, [&]() { cuda->sync(); }));
    cudaProfilerStop();
};
} // namespace infini
