#include "code_gen/code_engine.h"
#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/search_engine.h"
#include "code_gen/tensor.h"
#include "test.h"

TEST(SAMPLE_GRAPH_2, Cuda_codeGenerate) {
    //                                 /->conv1x3->relu--\.
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

    // op1->addSuccessors(op2);
    // op2->setSuccessors({op3, op9});
    // op3->addSuccessors(op4);
    // op4->setSuccessors({op5, op7});
    // op5->addSuccessors(op6);
    // op6->addSuccessors(op15);
    // op7->addSuccessors(op8);
    // op8->addSuccessors(op15);
    // op9->addSuccessors(op10);
    // op10->setSuccessors({op11, op13});
    // op11->addSuccessors(op12);
    // op12->addSuccessors(op15);
    // op13->addSuccessors(op14);
    // op14->addSuccessors(op15);
    // op15->addSuccessors(op16);

    // op2->addPredecessors(op1);
    // op3->addPredecessors(op2);
    // op4->addPredecessors(op3);
    // op5->addPredecessors(op4);
    // op6->addPredecessors(op5);
    // op7->addPredecessors(op4);
    // op8->addPredecessors(op7);
    // op9->addPredecessors(op2);
    // op10->addPredecessors(op9);
    // op11->addPredecessors(op10);
    // op12->addPredecessors(op11);
    // op13->addPredecessors(op10);
    // op14->addPredecessors(op13);
    // op15->setPredecessors({op6, op8, op12, op14});
    // op16->addPredecessors(op15);

    // i0->setInputOf({op1});
    // i1->setInputOf({op2});
    // i2->setInputOf({op3, op9});
    // i3->setInputOf({op4});
    // i4->setInputOf({op5, op7});
    // i5->setInputOf({op6});
    // i6->setInputOf({op15});
    // i7->setInputOf({op8});
    // i8->setInputOf({op15});
    // i9->setInputOf({op10});
    // i10->setInputOf({op11, op13});
    // i11->setInputOf({op12});
    // i12->setInputOf({op15});
    // i13->setInputOf({op14});
    // i14->setInputOf({op15});
    // i15->setInputOf({op16});

    // i1->setOutputOf({op1});
    // i2->setOutputOf({op2});
    // i3->setOutputOf({op3});
    // i4->setOutputOf({op4});
    // i5->setOutputOf({op5});
    // i6->setOutputOf({op6});
    // i7->setOutputOf({op7});
    // i8->setOutputOf({op8});
    // i9->setOutputOf({op9});
    // i10->setOutputOf({op10});
    // i11->setOutputOf({op11});
    // i12->setOutputOf({op12});
    // i13->setOutputOf({op13});
    // i14->setOutputOf({op14});
    // i15->setOutputOf({op15});
    // i16->setOutputOf({op16});

    // w1->setInputOf({op1});
    // w3->setInputOf({op3});
    // w5->setInputOf({op5});
    // w7->setInputOf({op7});
    // w9->setInputOf({op9});
    // w11->setInputOf({op11});
    // w13->setInputOf({op13});
    // w16->setInputOf({op16});

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
    // g->addTensor(i11);
    // g->addTensor(i12);
    // g->addTensor(i13);
    // g->addTensor(i14);
    // g->addTensor(i15);
    // g->addTensor(i16);

    // g->addTensor(w1);
    // g->addTensor(w3);
    // g->addTensor(w5);
    // g->addTensor(w7);
    // g->addTensor(w9);
    // g->addTensor(w11);
    // g->addTensor(w13);
    // g->addTensor(w16);

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
    // g->addOp(op11);
    // g->addOp(op12);
    // g->addOp(op13);
    // g->addOp(op14);
    // g->addOp(op15);
    // g->addOp(op16);

    g->setInputs({i0});
    g->setOutputs({i16});

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
