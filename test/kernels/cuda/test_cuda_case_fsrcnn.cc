#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/bias2prelu.h"
#include "operators/conv.h"
#include "operators/conv2dreduce.h"
#include "operators/element_wise.h"
#include "operators/matmul.h"
#include "operators/reshape.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

Graph createGraph(Ref<CudaRuntimeObj> cuda, int batchSize) {
    Graph g = make_ref<GraphObj>(cuda);
    auto input = g->addTensor({batchSize, 32, 32, 1}); // NCHW
    // auto input = g->addTensor({16, 32, 32, 1}); // change to NHWC format
    vector<tuple<string, int, int, bool>> configs = {
        {"Conv", 56, 5, true},  {"Conv", 12, 1, true},
        {"Conv", 12, 3, false}, {"Conv", 12, 3, false},
        {"Conv", 12, 3, false}, {"Conv", 12, 3, true},
        {"Conv", 56, 1, true},  {"ConvTranposed", 1, 9, false}};
    auto x = input;
    for (auto &[op, f, r, pRelu] : configs) {
        if (r == 5 && op == "Conv") { // for the first conv
            auto w = g->addTensor({f, x->getDims()[3], r, r});
            x = g->addOp<ConvObj>(x, w, nullptr, r / 2, r / 2, 1, 1, 1, 1,
                                  nullptr, ActType::None, true)
                    ->getOutput();
            // if (pRelu) {
            //     // TODO: Conv_nhwc + Bias+PRelu
            //     // Alternative: Conv_nchw + Transpose(NCHW->NHWC)+Bias+PRelu
            //     x = g->addOp<ReluObj>(x, nullptr)->getOutput();
            // }
            auto bias = g->addTensor({x->getDims()[3]});
            x = g->addOp<BiasPReLU>(x, bias, nullptr, pRelu, 0.1)->getOutput();
            continue;
        }

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
            x = g->addOp<Conv2dReduce>(x, bias, nullptr, pRelu, 0.1, r / 2,
                                       r / 2)
                    ->getOutput();
        } else if (op == "ConvTranposed") {
            IT_ASSERT(r == 9);
            // x = g->addOp<ConvTransposed2dObj>(x, w, nullptr, 3, 3, 4, 4, 1,
            // 1,
            //                                   1, 1)
            //         ->getOutput();
            x = g->addOp<Conv2dReduceTranspose>(x, bias, nullptr, pRelu, 0.1, 3,
                                                3, 4, 4)
                    ->getOutput();
        } else
            IT_ASSERT(false);
    }
    g->print();
    g->dataMalloc();
    cuda->run(g, true);
    cuda->getPerfTime(g, true);
    return g;
};

TEST(Case, fsrcnn_direct_run) {
    auto cuda = make_ref<CudaRuntimeObj>();
    for (int batch : {1, 16}) {
        auto g = createGraph(cuda, batch);
        cudaProfilerStart();
        printf("E2E time (batch size %d) %.3lf\n", batch,
               timeit([&]() { cuda->runWithoutSync(g); },
                      [&]() { cuda->sync(); }));
        cudaProfilerStop();
    }
};

TEST(Case, fsrcnn_cuda_graph) {
    auto cuda = make_ref<CudaRuntimeObj>();
    for (int batch : {1, 16}) {
        auto g = createGraph(cuda, batch);

        cudaGraph_t graph;
        cudaGraphExec_t instance;

        checkCudaError(cudaDeviceSynchronize());
        cudaStream_t stream = cuda->getStream();
        // cudaStreamCaptureStatus log;
        // checkCudaError(cudaStreamIsCapturing(stream, &log));
        // printf("cudaStreamCaptureStatus %d\n", log);
        checkCudaError(
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        cuda->runWithoutSync(g);
        checkCudaError(cudaStreamEndCapture(stream, &graph));
        checkCudaError(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));

        cudaProfilerStart();
        printf(
            "CUDA graph time (batch size %d): %.3lf ms\n", batch,
            timeit([&]() { checkCudaError(cudaGraphLaunch(instance, stream)); },
                   [&]() { cudaStreamSynchronize(stream); }));
        cudaProfilerStop();
    }
};
} // namespace infini
