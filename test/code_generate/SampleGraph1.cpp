#include "code_gen/code_engine.h"
#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/search_engine.h"
#include "code_gen/tensor.h"
#include "test.h"

TEST(SAMPLE_GRAPH_1, Cuda_codeGenerate) {
    // conv7x7->relu->conv3x3->relu->conv3x3->relu->conv3x3->relu->conv3x3->relu
    auto g = new tpm::Graph();
    auto i0 = g->tensor({16, 3, 224, 224});
    auto i1 = g->tensor({16, 64, 56, 56});
    auto i2 = g->tensor({16, 64, 56, 56});
    auto i3 = g->tensor({16, 128, 28, 28});
    auto i4 = g->tensor({16, 128, 28, 28});
    auto i5 = g->tensor({16, 128, 28, 28});
    auto i6 = g->tensor({16, 128, 28, 28});
    auto i7 = g->tensor({16, 256, 14, 14});
    auto i8 = g->tensor({16, 256, 14, 14});
    auto i9 = g->tensor({16, 256, 14, 14});
    auto i10 = g->tensor({16, 256, 14, 14});

    auto w1 = g->tensor({64, 3, 7, 7});
    auto w3 = g->tensor({128, 64, 3, 3});
    auto w5 = g->tensor({128, 128, 3, 3});
    auto w7 = g->tensor({256, 128, 3, 3});
    auto w9 = g->tensor({256, 256, 3, 3});

    g->conv(i0, w1, i1, 3, 3, 4, 4);
    g->relu(i1, i2);
    g->conv(i2, w3, i3, 1, 1, 2, 2);
    g->relu(i3, i4);
    g->conv(i4, w5, i5, 1, 1, 1, 1);
    g->relu(i5, i6);
    g->conv(i6, w7, i7, 1, 1, 2, 2);
    g->relu(i7, i8);
    g->conv(i8, w9, i9, 1, 1, 1, 1);
    g->relu(i9, i10);

    // op1->addSuccessors(op2);
    // op2->addSuccessors(op3);
    // op3->addSuccessors(op4);
    // op4->addSuccessors(op5);
    // op5->addSuccessors(op6);
    // op6->addSuccessors(op7);
    // op7->addSuccessors(op8);
    // op8->addSuccessors(op9);
    // op9->addSuccessors(op10);

    // op2->addPredecessors(op1);
    // op3->addPredecessors(op2);
    // op4->addPredecessors(op3);
    // op5->addPredecessors(op4);
    // op6->addPredecessors(op5);
    // op7->addPredecessors(op6);
    // op8->addPredecessors(op7);
    // op9->addPredecessors(op8);
    // op10->addPredecessors(op9);

    // i0->addInputOf(op1);
    // i1->addInputOf(op2);
    // i2->addInputOf(op3);
    // i3->addInputOf(op4);
    // i4->addInputOf(op5);
    // i5->addInputOf(op6);
    // i6->addInputOf(op7);
    // i7->addInputOf(op8);
    // i8->addInputOf(op9);
    // i9->addInputOf(op10);

    // i1->setOutputOf(op1);
    // i2->setOutputOf(op2);
    // i3->setOutputOf(op3);
    // i4->setOutputOf(op4);
    // i5->setOutputOf(op5);
    // i6->setOutputOf(op6);
    // i7->setOutputOf(op7);
    // i8->setOutputOf(op8);
    // i9->setOutputOf(op9);
    // i10->setOutputOf(op10);

    // w1->addInputOf(op1);
    // w3->addInputOf(op3);
    // w5->addInputOf(op5);
    // w7->addInputOf(op7);
    // w9->addInputOf(op9);

    // g->addTensor(i0);
    // g->addTensor(i1);
    // g->addTensor(i2);
    // g->addTensor(i3);
    // g->addTensor(i4);
    // g->addTensor(i5);
    // g->addTensor(i6);
    // g->addTensor(i7);
    // g->addTensor(i8);
    // g->addTensor(i9);
    // g->addTensor(i10);

    // g->addTensor(w1);
    // g->addTensor(w3);
    // g->addTensor(w5);
    // g->addTensor(w7);
    // g->addTensor(w9);

    // g->addOp(op1);
    // g->addOp(op2);
    // g->addOp(op3);
    // g->addOp(op4);
    // g->addOp(op5);
    // g->addOp(op6);
    // g->addOp(op7);
    // g->addOp(op8);
    // g->addOp(op9);
    // g->addOp(op10);

    g->updateConnection();

    std::shared_ptr<tpm::SubGraph> graph, bestGraph;
    graph = std::make_shared<tpm::SubGraph>(g->getOperators());
    tpm::SearchEngine searchEngine(std::make_shared<tpm::Generator>());
    searchEngine.run(graph, bestGraph);
    tpm::CodeEngine codeEngine;
    auto perfEngine = searchEngine.exportPerfEngine();
    codeEngine.importPerfEngine(perfEngine);
    codeEngine.genCode(bestGraph, "res.cu");
}
