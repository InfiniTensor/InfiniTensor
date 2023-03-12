#include "core/blob.h"
#include "core/dummy_mutator.h"
#include "core/graph.h"
#include "core/runtime.h"
#include "core/search_engine.h"
#include "cuda/cuda_runtime.h"
#include "nnet/nmutator.h"
#include "operators/conv.h"
#include "test.h"

namespace infini {

TEST(Mutator, NaiveConvWithInterpreter) {
    // verifyNaiveMembound True: subgraph after transformation
    // verifyNaiveMembound False: subgraph of one single membound (eOP)
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    // const bool verifyNaiveMembound = false;

    auto i0 = g->addTensor({1, 3, 32, 32}, DataType::UInt32);
    auto w1 = g->addTensor({2, 3, 3, 3}, DataType::UInt32);
    g->addOp<ConvObj>(i0, w1, nullptr, 1, 1);
    printf("--- Init Finished ---\n");

    auto mutator = make_ref<NMutator>();
    mutator->setToNaiveMembound();
    SearchEngine searchEngine(runtime, mutator);
    // g->dataMalloc();
    auto bestGraph = searchEngine.run(g);
    bestGraph->print();
    printf("--- SearchEngine Finished ---\n");

    auto mutatedGraphs = mutator->run(g);
    IT_ASSERT(mutatedGraphs.size() == 2);
    printf("--- Mutator Finished ---\n");

    auto gg = mutatedGraphs[1];
    g->dataMalloc();
    gg->dataMalloc();
    for (auto t : g->getTensors()) {
        if (t->getFuid() <= 2)
            t->setData(IncrementalGenerator());
    }
    for (auto t : gg->getTensors()) {
        if (t->getFuid() <= 2)
            t->setData(IncrementalGenerator());
    }
    runtime->run(g);
    runtime->run(gg);
    gg->print();

    EXPECT_TRUE(g->getOutputs()[0]->equalData(gg->getOutputs()[0]));
    EXPECT_TRUE(g->getOutputs()[0]->getRawDataPtr<void *>() !=
                gg->getOutputs()[0]->getRawDataPtr<void *>());
}

// FIXME: failed since implicit transpose for DLT
TEST(Mutator, InfoGAN_TConv_3_correctness) {
    // verifyNaiveMembound True: subgraph after transformation
    // verifyNaiveMembound False: subgraph of one single membound (eOP)
    // const bool verifyNaiveMembound = false;
    Runtime runtime = make_ref<CudaRuntimeObj>();
    Graph g = make_ref<GraphObj>(runtime);
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);

    // TODO: recover me for InfoGAN
    const int n = 1, c = 256, h = 2, w = 2, f = 448, r = 4, s = 4;
    // // Minimum config for test
    // const int n = 1, c = 1, h = 2, w = 2, f = 1, r = 4, s = 4;
    auto i0 = g->addTensor({n, h, w, f});
    auto w0 = g->addTensor({f, r, s, c});
    g->addOp<ConvTransposed2dNHWCObj>(i0, w0, nullptr, 1, 1, 2, 2, 1, 1);
    g->print();

    auto mutator =
        make_ref<NMutator>(NMutator::Mode::RuleBased,
                           vector<int>{3, 2, 2, 2, 2, 5, 8, 8, 6, 91, 90});
    // mutator->setToNaiveMembound();
    SearchEngine searchEngine(runtime, mutator);
    auto bestGraph = searchEngine.run(g);
    bestGraph->print();
    printf("--- SearchEngine Finished ---\n");

    g->dataMalloc();
    bestGraph->dataMalloc();
    for (auto t : g->getInputs()) {
        t->setData(IncrementalGenerator());
    }
    for (auto t : bestGraph->getInputs()) {
        t->setData(IncrementalGenerator());
    }
    for (auto t : g->getOutputs()) {
        t->setData(IncrementalGenerator());
    }
    for (auto t : bestGraph->getOutputs()) {
        t->setData(IncrementalGenerator());
    }
    runtime->run(g);
    // puts("cuDNN");
    // g->getOutputs()[0]->printData();
    runtime->run(bestGraph);
    // puts("Output");
    // bestGraph->getOutputs()[0]->printData();

    auto go0 = gCpu->cloneTensor(g->getOutputs()[0]);
    auto bgo0 = gCpu->cloneTensor(bestGraph->getOutputs()[0]);

    EXPECT_TRUE(go0->equalData(bgo0));
    EXPECT_TRUE(g->getOutputs()[0]->getRawDataPtr<void *>() !=
                bestGraph->getOutputs()[0]->getRawDataPtr<void *>());
}

// TEST(Mutator, Conv9x9) {
//     auto g = new tpm::Graph();
//     auto i0 = g->tensor({1, 1, 224, 224});

//     auto w1 = g->tensor({64, 1, 9, 9});

//     g->conv(i0, w1, 4, 4);

//     g->updateConnection();

//     std::shared_ptr<tpm::SubGraph> graph, bestGraph;
//     graph = std::make_shared<tpm::SubGraph>(g->getOperators());
//     tpm::SearchEngine<tpm::NMutator> searchEngine;
//     searchEngine.run(graph, bestGraph);
//     tpm::CodeEngine codeEngine;
//     auto perfEngine = searchEngine.exportPerfEngine();
//     codeEngine.importPerfEngine(perfEngine);
//     codeEngine.genCode(bestGraph, "res.cu");
// }

// TEST(Mutator, TConv_1) {
//     auto g = new tpm::Graph();

//     auto i0 = g->tensor({1, 1, 1, 228});
//     auto w1 = g->tensor({228, 2, 2, 448});

//     // g->conv(i0, w1, 4, 4);
//     g->convTrans(i0, w1, 0, 0, 1, 1);

//     g->updateConnection();

//     std::shared_ptr<tpm::SubGraph> graph, bestGraph;
//     graph = std::make_shared<tpm::SubGraph>(g->getOperators());
//     tpm::SearchEngine<tpm::NMutator> searchEngine;
//     searchEngine.run(graph, bestGraph);
//     tpm::CodeEngine codeEngine;
//     auto perfEngine = searchEngine.exportPerfEngine();
//     codeEngine.importPerfEngine(perfEngine);
//     codeEngine.genCode(bestGraph, "res.cu");
// }

// TEST(Mutator, TConv_3) {
//     auto g = new tpm::Graph();

//     auto i0 = g->tensor({1, 2, 2, 448});
//     auto w1 = g->tensor({448, 4, 4, 256});

//     // g->conv(i0, w1, 4, 4);
//     g->convTrans(i0, w1, 1, 1, 2, 2, 1, 1);

//     g->updateConnection();

//     std::shared_ptr<tpm::SubGraph> graph, bestGraph;
//     graph = std::make_shared<tpm::SubGraph>(g->getOperators());
//     tpm::SearchEngine<tpm::NMutator> searchEngine;
//     searchEngine.run(graph, bestGraph);
//     tpm::CodeEngine codeEngine;
//     auto perfEngine = searchEngine.exportPerfEngine();
//     codeEngine.importPerfEngine(perfEngine);
//     codeEngine.genCode(bestGraph, "res.cu");
// }

// TEST(Mutator, G2BMM) {
//     auto g = new tpm::Graph();

//     int nHeads = 8, seq_len = 10000, feat_len = 64, w = 1000, d = 4;
//     auto i0 = g->tensor({nHeads, seq_len, feat_len});
//     auto i1 = g->tensor({nHeads, seq_len, feat_len});

//     g->g2bmm(i0, i1, w, d);

//     g->updateConnection();

//     std::shared_ptr<tpm::SubGraph> graph, bestGraph;
//     graph = std::make_shared<tpm::SubGraph>(g->getOperators());
//     tpm::SearchEngine<tpm::NMutator> searchEngine;
//     searchEngine.run(graph, bestGraph);
//     tpm::CodeEngine codeEngine;
//     auto perfEngine = searchEngine.exportPerfEngine();
//     codeEngine.importPerfEngine(perfEngine);
//     codeEngine.genCode(graph, "res.cu");
//     // codeEngine.genCode(bestGraph, "res.cu");
// }

// TEST(Mutator, GBMML) {
//     auto g = new tpm::Graph();

//     int nHeads = 8, seq_len = 10000, feat_len = 64, w = 1000, d = 4;
//     auto i0 = g->tensor({nHeads, seq_len, 2 * w + 1});
//     auto i1 = g->tensor({nHeads, seq_len, feat_len});

//     g->gbmml(i0, i1, d);

//     g->updateConnection();

//     std::shared_ptr<tpm::SubGraph> graph, bestGraph;
//     graph = std::make_shared<tpm::SubGraph>(g->getOperators());
//     tpm::SearchEngine<tpm::NMutator> searchEngine;
//     searchEngine.run(graph, bestGraph);
//     tpm::CodeEngine codeEngine;
//     auto perfEngine = searchEngine.exportPerfEngine();
//     codeEngine.importPerfEngine(perfEngine);
//     codeEngine.genCode(graph, "res.cu");
//     // codeEngine.genCode(bestGraph, "res.cu");
// }

// TEST(Mutator, Conv5x5) {
//     //
//     conv7x7->relu->conv3x3->relu->conv3x3->relu->conv3x3->relu->conv3x3->relu
//     auto g = new tpm::Graph();
//     auto i0 = g->tensor({1, 32, 224, 224});

//     auto w1 = g->tensor({1, 32, 5, 5});

//     g->conv(i0, w1, tpm::ConvOp::PaddingMode::Same);

//     g->updateConnection();

//     std::shared_ptr<tpm::SubGraph> graph, bestGraph;
//     graph = std::make_shared<tpm::SubGraph>(g->getOperators());
//     tpm::SearchEngine<tpm::NMutator> searchEngine;
//     searchEngine.run(graph, bestGraph);
//     tpm::CodeEngine codeEngine;
//     auto perfEngine = searchEngine.exportPerfEngine();
//     codeEngine.importPerfEngine(perfEngine);
//     codeEngine.genCode(bestGraph, "res.cu");
// }

// TEST(Mutator, BMM) {
//     const int m = 16, n = 1024, k = 1024;
//     auto g = new tpm::Graph();
//     auto i0 = g->tensor({1, m, k});
//     auto w0 = g->tensor({1, k, n});
//     auto w1 = g->tensor({1, k, n});
//     auto w2 = g->tensor({1, k, n});

//     g->matmul(i0, w0);
//     g->matmul(i0, w1);
//     g->matmul(i0, w2);
//     g->updateConnection();

//     std::shared_ptr<tpm::SubGraph> graph, bestGraph;
//     graph = std::make_shared<tpm::SubGraph>(g->getOperators());
//     tpm::SearchEngine<tpm::NMutator> searchEngine;
//     searchEngine.run(graph, bestGraph);
//     tpm::CodeEngine codeEngine;
//     auto perfEngine = searchEngine.exportPerfEngine();
//     codeEngine.importPerfEngine(perfEngine);
//     codeEngine.genCode(bestGraph, "res.cu");
// }

// TEST(Mutator, Conv2gemm1x1_bs1_mutator) {
//     const int N = 1, H = 7, W = 7, C = 512, F = 512, R = 1, S = 1;
//     auto g = new tpm::Graph();
//     auto i0 = g->tensor({N, C, H, W});
//     auto w1 = g->tensor({F, C, R, S});
//     g->conv(i0, w1, R / 2, S / 2);

//     g->updateConnection();

//     std::shared_ptr<tpm::SubGraph> graph, bestGraph;
//     vector<tpm::SubGraph *> out_graphs;
//     graph = std::make_shared<tpm::SubGraph>(g->getOperators());
//     const vector<int> rules = {3, 2, 2, 8, 8, 6, 6};
//     auto mutator = make_shared<tpm::NMutator>(rules);
//     mutator->run(graph.get(), out_graphs);
//     tpm::SearchEngine searchEngine(mutator);
//     int maxNReshapes = 0;
//     for (const auto &graph : out_graphs) {
//         searchEngine.getPerf(make_shared<tpm::SubGraph>(*graph), true);
//         int nReshapes = 0, nTrans = 0;
//         for (auto op : graph->getOperators()) {
//             nReshapes += op->isReshapeOp();
//             if (auto matmul = dynamic_cast<MatmulOp *>(op))
//                 nTrans = matmul->getTransA() + matmul->getTransB();
//         }
//         maxNReshapes = max(maxNReshapes, nReshapes);
//         // Number of Reshapes for KxA and AxK
//         EXPECT_TRUE((nReshapes == 3 - nTrans) || (nReshapes == nTrans));
//     }
//     // Matmul K^N A^N -> no Membound
//     EXPECT_EQ(maxNReshapes, 3);
// }

// TEST(Mutator, Conv2gemm1x1_searchEngine_ruleBased) {
//     const int N = 1, H = 7, W = 7, C = 512, F = 512, R = 1, S = 1;
//     auto g = new tpm::Graph();
//     auto i0 = g->tensor({N, C, H, W});
//     auto w1 = g->tensor({F, C, R, S});
//     g->conv(i0, w1, R / 2, S / 2);

//     g->updateConnection();

//     std::shared_ptr<tpm::SubGraph> graph, bestGraph;
//     graph = std::make_shared<tpm::SubGraph>(g->getOperators());
//     const vector<int> rules = {3, 2, 2, 8, 8, 6, 6};
//     tpm::SearchEngine searchEngine(make_shared<tpm::NMutator>(rules));
//     searchEngine.run(graph, bestGraph);

//     // clang-format off
//     // ========== PET graph getPerf ============
//     // Reshape(in=0,out=126)
//     //  op_time 0.000000
//     // Reshape(in=1,out=125)
//     //  op_time 0.000000
//     // Matmul([A,B,act=0],A=125,B=126,C=124, TTbmnk: 0, 0, 1, 512, 49, 512)
//     //  op_time 0.013799
//     // Reshape(in=124,out=3)
//     //  op_time 0.000000
//     //          Op Cnt   T_tot Percent  T_mean
//     //      Matmul   1   0.014   100.0   0.014
//     //     Reshape   3   0.000     0.0   0.000
//     // Origin Perf: 0.0553319
//     // Best Perf without correction: 0.0137989
//     // Best Perf with correction: 0.0137989
//     // clang-format on
//     EXPECT_EQ(bestGraph->getOperators().size(), 4u);
//     auto cntOps = bestGraph->countOps();
//     EXPECT_EQ(cntOps["Matmul"], 1);
//     EXPECT_EQ(cntOps["Reshape"], 3);
//     bestGraph->print();
// }

// TEST(Mutator, Conv2gemm1x1_searchEngine_search) {
//     const int N = 1, H = 7, W = 7, C = 512, F = 512, R = 1, S = 1;
//     auto g = new tpm::Graph();
//     auto i0 = g->tensor({N, C, H, W});
//     auto w1 = g->tensor({F, C, R, S});
//     g->conv(i0, w1, R / 2, S / 2);

//     g->updateConnection();

//     std::shared_ptr<tpm::SubGraph> graph, bestGraph;
//     graph = std::make_shared<tpm::SubGraph>(g->getOperators());
//     tpm::SearchEngine searchEngine(make_shared<tpm::NMutator>());
//     searchEngine.run(graph, bestGraph);

//     EXPECT_EQ(bestGraph->getOperators().size(), 4u);
//     auto cntOps = bestGraph->countOps();
//     EXPECT_EQ(cntOps["Matmul"], 1);
//     EXPECT_EQ(cntOps["Reshape"], 3);
//     bestGraph->print();
// }

// TEST(Mutator, Conv2gemm1x7_searchEngine_ruleBased) {
//     const int N = 1, C = 2048, H = 7, W = 7, F = 128, R = 1,
//               S = 7; // gcn_Conv_137
//     auto g = new tpm::Graph();
//     auto i0 = g->tensor({N, C, H, W});
//     auto w1 = g->tensor({F, C, R, S});
//     g->conv(i0, w1, R / 2, S / 2);

//     g->updateConnection();

//     std::shared_ptr<tpm::SubGraph> graph, bestGraph;
//     graph = std::make_shared<tpm::SubGraph>(g->getOperators());
//     const vector<int> rules = {3, 2, 2, 5, 8, 8, 6, 90};
//     tpm::SearchEngine searchEngine(make_shared<tpm::NMutator>(rules));
//     searchEngine.run(graph, bestGraph);

//     // clang-format off
//     //     ========== PET graph getPerf ============
//     // Reshape(in=0,out=309)
//     //  op_time 0.000000
//     // MemBound[124644277](i0=1, o0=308, exec_time=0.0683594, NNet
//     Inputs=[K,])
//     // L<c:0:2048><i52:0:896>Sum  ...  [i52,c]
//     //     {L<i52:0:896><c:0:2048>Sum  ...  [(i52 / 7),c,((i52 / 7) % 1),(i52
//     % 7)]
//     //     {K}}

//     //  op_time 0.000000
//     // Matmul([A^T,B,act=0],A=308,B=309,C=307, TTbmnk: 1, 0, 1, 896, 49,
//     2048)
//     //  op_time 0.024471
//     // MemBound[124644277](i0=307, o0=3, exec_time=0.001, NNet Inputs=[T49,])
//     // L<n:0:1><f:0:128><h:0:7><w:0:7>Sum<r:0:1><s:0:7>  ...  [(h + r),r,(w +
//     s),s,n,f]
//     //
//     {L<i45:0:7><i46:0:1><i26:3:10><i27:0:7><n:0:1><f:0:128><pad=0,0,3,0,0,0,>Sum
//     ...  [(((7 * f) + (7 * i46)) + i27),(((49 * n) + (7 * i45)) + (i26 +
//     -3))]
//     //     {T49}}

//     //  op_time 0.001000
//     //          Op Cnt   T_tot Percent  T_mean
//     //      Matmul   1   0.024    96.1   0.024
//     //     Reshape   1   0.000     0.0   0.000
//     //    MemBound   2   0.001     3.9   0.001
//     // Origin Perf: 0.405595
//     // Best Perf without correction: 0.0254715
//     // Best Perf with correction: 0.0254715
//     // Transpose perf: 0
//     // clang-format on
//     EXPECT_EQ(bestGraph->getOperators().size(), 4u);
//     auto cntOps = bestGraph->countOps();
//     EXPECT_EQ(cntOps["Matmul"], 1);
//     EXPECT_EQ(cntOps["Reshape"], 1);
//     EXPECT_EQ(cntOps["MemBound"], 2);
//     bestGraph->print();
//     EXPECT_TRUE(graph->verification(bestGraph.get(), true));
// }

// TEST(Mutator, Conv2gemm7x1_searchEngine_ruleBased) {
//     const int N = 1, C = 2048, H = 7, W = 7, F = 128, R = 7,
//               S = 1; // gcn_Conv_137
//     auto g = new tpm::Graph();
//     auto i0 = g->tensor({N, C, H, W});
//     auto w1 = g->tensor({F, C, R, S});
//     g->conv(i0, w1, R / 2, S / 2);

//     g->updateConnection();

//     std::shared_ptr<tpm::SubGraph> graph, bestGraph;
//     graph = std::make_shared<tpm::SubGraph>(g->getOperators());
//     const vector<int> rules = {3, 2, 2, 5, 8, 8, 6, 90};
//     tpm::SearchEngine searchEngine(make_shared<tpm::NMutator>(rules));
//     searchEngine.run(graph, bestGraph);

//     EXPECT_EQ(bestGraph->getOperators().size(), 4u);
//     auto cntOps = bestGraph->countOps();
//     EXPECT_EQ(cntOps["Matmul"], 1);
//     EXPECT_EQ(cntOps["Reshape"], 1);
//     EXPECT_EQ(cntOps["MemBound"], 2);
//     bestGraph->print();
//     EXPECT_TRUE(graph->verification(bestGraph.get(), true));
// }

// TEST(Mutator, Conv2gemm7x1_searchEngine_search) {
//     const int N = 1, C = 2048, H = 7, W = 7, F = 128, R = 7,
//               S = 1; // gcn_Conv_137
//     auto g = new tpm::Graph();
//     auto i0 = g->tensor({N, C, H, W});
//     auto w1 = g->tensor({F, C, R, S});
//     g->conv(i0, w1, R / 2, S / 2);

//     g->updateConnection();

//     std::shared_ptr<tpm::SubGraph> graph, bestGraph;
//     graph = std::make_shared<tpm::SubGraph>(g->getOperators());
//     // const vector<int> rules = {3, 2, 2, 5, 8, 8, 6, 90};
//     tpm::SearchEngine searchEngine(make_shared<tpm::NMutator>());
//     searchEngine.run(graph, bestGraph);

//     EXPECT_EQ(bestGraph->getOperators().size(), 4u);
//     auto cntOps = bestGraph->countOps();
//     EXPECT_EQ(cntOps["Matmul"], 1);
//     EXPECT_EQ(cntOps["Reshape"], 1);
//     EXPECT_EQ(cntOps["MemBound"], 2);
//     bestGraph->print();
//     EXPECT_TRUE(graph->verification(bestGraph.get(), true));
// }
} // namespace infini
