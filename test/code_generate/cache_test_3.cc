#include "code_gen/cmutator.h"
#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/search_engine.h"
#include "code_gen/tensor.h"
#include "test.h"

TEST(CACHE_TEST_3, Cuda_codeGenerate) {    //                                 /->conv1x3->relu--\.
    // conv3x3->relu---->conv1x1->relu--->conv3x1->relu---->concat->conv3x3
    //                \->conv1x1->relu--->conv3x1->relu--/
    //                                 \->conv1x3->relu-/
    auto g = new tpm::Graph();
    auto i0 = g->tensor({8, 64, 28, 28});
    auto i1 = g->tensor({8, 256, 14, 14});
    auto i2 = g->tensor({8, 256, 14, 14});
    auto i3 = g->tensor({8, 384, 14, 14});
    auto i4 = g->tensor({8, 384, 14, 14});
    auto i5 = g->tensor({8, 384, 14, 14});
    auto i6 = g->tensor({8, 384, 14, 14});
    auto i7 = g->tensor({8, 384, 14, 14});
    auto i8 = g->tensor({8, 384, 14, 14});
    auto i9 = g->tensor({8, 128, 14, 14});
    auto i10 = g->tensor({8, 128, 14, 14});
    auto i11 = g->tensor({8, 128, 14, 14});
    auto i12 = g->tensor({8, 128, 14, 14});
    auto i13 = g->tensor({8, 128, 14, 14});
    auto i14 = g->tensor({8, 128, 14, 14});
    auto i15 = g->tensor({8, 1024, 14, 14});
    auto i16 = g->tensor({8, 1024, 7, 7});

    auto w1 = g->tensor({256, 64, 3, 3});
    auto w3 = g->tensor({384, 256, 1, 1});
    auto w5 = g->tensor({384, 384, 1, 3});
    auto w7 = g->tensor({384, 384, 3, 1});
    auto w9 = g->tensor({128, 256, 1, 1});
    auto w11 = g->tensor({128, 128, 1, 3});
    auto w13 = g->tensor({128, 128, 3, 1});
    auto w16 = g->tensor({1024, 1024, 3, 3});

    g->conv(i0, w1, i1, 1, 1, 2, 2);
    g->relu(i1, i2);

    g->conv(i2, w3, i3, 0, 0);
    g->relu(i3, i4);
    g->conv(i4, w5, i5, 0, 1);
    g->relu(i5, i6);
    g->conv(i4, w7, i7, 1, 0);
    g->relu(i7, i8);

    g->conv(i2, w9, i9, 0, 0);
    g->relu(i9, i10);
    g->conv(i10, w11, i11, 0, 1);
    g->relu(i11, i12);
    g->conv(i10, w13, i13, 1, 0);
    g->relu(i13, i14);

    g->concat({i6, i8, i12, i14}, i15, 1);
    g->conv(i15, w16, i16, 1, 1, 2, 2);

    g->setInputs({i0});
    g->setOutputs({i16});

    g->updateConnection();

    std::shared_ptr<tpm::SubGraph> graph, bestGraph;
    graph = std::make_shared<tpm::SubGraph>(g->getOperators());
    tpm::SearchEngine searchEngine(std::make_shared<tpm::CMutator>());
    searchEngine.run(graph, bestGraph);

    std::cout << "Graph:" << std::endl;
    graph->print();
    std::cout << "BestGraph:" << std::endl;
    bestGraph->print();
}