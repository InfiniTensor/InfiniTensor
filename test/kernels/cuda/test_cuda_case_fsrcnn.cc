#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/conv.h"
#include "operators/conv2dreduce.h"
#include "operators/matmul.h"
#include "operators/reshape.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {
TEST(Case, fsrcnn) {
    auto cuda = make_ref<CudaRuntimeObj>();
    Graph g = make_ref<GraphObj>(cuda);
    // auto input = g->addTensor({1, 1, 32, 32});
    auto input = g->addTensor({16, 32, 32, 1}); // change to (n, h, w, c) format
    vector<tuple<string, int, int, bool>> configs = {
        {"Conv", 56, 5, true},  {"Conv", 12, 1, true},
        {"Conv", 12, 3, false}, {"Conv", 12, 3, false},
        {"Conv", 12, 3, false}, {"Conv", 12, 3, true},
        {"Conv", 56, 1, true},  {"ConvTranposed", 56, 9, false}};
    auto x = input;
    for (auto &[op, f, r, pRelu] : configs) {
        auto idim = x->getDims();
        int n = idim[0], h = idim[1], w = idim[2], c = idim[3];
        x = g->addOp<ReshapeObj>(x, nullptr, Shape{1, n * h * w, c})
                ->getOutput();
        auto weight = g->addTensor({1, x->getDims()[2], f * r * r});
        x = g->addOp<MatmulObj>(x, weight, nullptr)->getOutput();
        x = g->addOp<ReshapeObj>(x, nullptr, Shape{n, h, w, f, r, r})
                ->getOutput();
        auto bias = g->addTensor({f});
        if (op == "Conv") {
            x = g->addOp<Conv2dReduce>(x, bias, nullptr, pRelu, r / 2, r / 2)
                    ->getOutput();
        } else if (op == "ConvTranposed") {
            IT_ASSERT(r == 9);
            // x = g->addOp<ConvTransposed2dObj>(x, w, nullptr, 3, 3, 4, 4, 1,
            // 1,
            //                                   1, 1)
            //         ->getOutput();
            x = g->addOp<Conv2dReduceTranspose>(x, bias, nullptr, pRelu, 3, 3,
                                                4, 4)
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
