#include "code_engine.h"
#include "nnet/nmutator.h"
#include "operator.h"
#include "search_engine.h"
#include "tensor.h"
#include "gtest/gtest.h"
using namespace std;
namespace ch {
using namespace std::chrono;
}

TEST(Longformer, e2e_bs1_depth) {
    const int bs = 1, seqlen = 10000, w = 1000, featlen = 512, heads = 8, d = 4;
    const int hidden = featlen, hiddenPerHead = hidden / heads;
    assert(hidden % heads == 0);
    auto g = new tpm::Graph();

    auto i0 = g->tensor({bs, seqlen, featlen});
    auto w0 = g->tensor({featlen, hidden});
    auto w1 = g->tensor({512, 512});
    auto w2 = g->tensor({512, 512});
    // Feed forward
    auto w3 = g->tensor({512, 512});
    auto bias3 = g->tensor({512});
    auto w4 = g->tensor({512, 512});
    auto bias4 = g->tensor({512});

    auto q0 = g->tensor({bs, seqlen, hidden});
    auto k0 = g->tensor({bs, seqlen, hidden});
    auto v0 = g->tensor({bs, seqlen, hidden});

    auto q1 = g->tensor({bs, seqlen, heads, hiddenPerHead});
    auto k1 = g->tensor({bs, seqlen, heads, hiddenPerHead});
    auto v1 = g->tensor({bs, seqlen, heads, hiddenPerHead});

    auto q2 = g->tensor({bs, heads, seqlen, hiddenPerHead});
    auto k2 = g->tensor({bs, heads, seqlen, hiddenPerHead});
    auto v2 = g->tensor({bs, heads, seqlen, hiddenPerHead});

    auto q3 = g->tensor({bs * heads, seqlen, hiddenPerHead});
    auto k3 = g->tensor({bs * heads, seqlen, hiddenPerHead});
    auto v3 = g->tensor({bs * heads, seqlen, hiddenPerHead});

    auto prob = g->tensor({bs * heads, seqlen, 2 * w + 1});
    auto probSoftmax = g->tensor({bs * heads, seqlen, 2 * w + 1});
    auto attn = g->tensor({bs * heads, seqlen, hiddenPerHead});

    auto t00 = g->tensor({bs, seqlen, hidden});
    auto t01 = g->tensor({bs, seqlen, hidden});
    auto t02 = g->tensor({bs, seqlen, hidden});
    // auto t10 = g->tensor({bs, seqlen, hidden});
    auto t11 = g->tensor({bs, seqlen, hidden});
    auto t12 = g->tensor({bs, seqlen, hidden});
    auto output = g->tensor({bs, seqlen, featlen});

    g->matmul(i0, w0, q0, false, true);
    g->matmul(i0, w1, k0, false, true);
    g->matmul(i0, w2, v0, false, true);
    g->reshape(q0, q1);
    g->reshape(k0, k1);
    g->reshape(v0, v1);
    g->transpose(q1, q2, 0, {{0, -1}, 2, 1, 3}, 1);
    g->transpose(k1, k2, 0, {{0, -1}, 2, 1, 3}, 1);
    g->transpose(v1, v2, 0, {{0, -1}, 2, 1, 3}, 1);
    g->reshape(q2, q3);
    g->reshape(k2, k3);
    g->reshape(v2, v3);
    // Attention
    g->g2bmm(q3, k3, prob, w, d);
    g->softmax(prob, probSoftmax, 2);
    g->gbmml(probSoftmax, v3, attn, d);
    g->transpose(attn, t00, 0, {0, 1, {-1, 2}}, heads);

    // Feed forward
    g->matmul(t00, w3, t01, false, true, bias3);
    g->relu(t01, t02);
    g->matmul(t02, w4, t11, false, true, bias4);
    g->relu(t11, t12);
    g->add({t12, i0}, output);

    g->updateConnection();

    for (int i = 0; i <= 8; ++i) {
        ch::time_point<ch::high_resolution_clock, ch::nanoseconds> beg, end;
        beg = ch::high_resolution_clock::now();
        std::shared_ptr<tpm::SubGraph> graph, bestGraph;
        graph = std::make_shared<tpm::SubGraph>(g->getOperators());
        auto mutationEngine = make_shared<tpm::NMutator>();
        mutationEngine->setMaxDepth(i);
        tpm::SearchEngine searchEngine(mutationEngine);
        searchEngine.run(graph, bestGraph);
        // dbg("bestGraph");
        bestGraph->print();
        tpm::CodeEngine codeEngine;
        // codeEngine.importPerfEngine(perfEngine);
        // codeEngine.genCode(bestGraph, "res.cu");

        // const auto originalTime = searchEngine.getPerf(graph, true);
        const auto bestTime = searchEngine.getPerf(bestGraph, true);
        // dbg(originalTime, bestTime);
        // EXPECT_GE(originalTime, 45);
        // EXPECT_LE(bestTime, 25);
        end = ch::high_resolution_clock::now();
        double t = ch::duration_cast<ch::duration<double>>(end - beg).count();
        // printf("====== maxdepth=%d \n", i);
        printf("Statistics: maxdepth %d , time %.3lf s, states %lld , "
               "candidate %lld , best time %lf\n",
               i, t, mutationEngine->cntStates, mutationEngine->cntCandidates,
               bestTime);
    }
}

TEST(Longformer, e2e_bs1) {
    const int bs = 1, seqlen = 10000, w = 1000, featlen = 512, heads = 8, d = 4;
    const int hidden = featlen, hiddenPerHead = hidden / heads;
    assert(hidden % heads == 0);
    auto g = new tpm::Graph();

    auto i0 = g->tensor({bs, seqlen, featlen});
    auto w0 = g->tensor({featlen, hidden});
    auto w1 = g->tensor({512, 512});
    auto w2 = g->tensor({512, 512});
    // Feed forward
    auto w3 = g->tensor({512, 512});
    auto bias3 = g->tensor({512});
    auto w4 = g->tensor({512, 512});
    auto bias4 = g->tensor({512});

    auto q0 = g->tensor({bs, seqlen, hidden});
    auto k0 = g->tensor({bs, seqlen, hidden});
    auto v0 = g->tensor({bs, seqlen, hidden});

    auto q1 = g->tensor({bs, seqlen, heads, hiddenPerHead});
    auto k1 = g->tensor({bs, seqlen, heads, hiddenPerHead});
    auto v1 = g->tensor({bs, seqlen, heads, hiddenPerHead});

    auto q2 = g->tensor({bs, heads, seqlen, hiddenPerHead});
    auto k2 = g->tensor({bs, heads, seqlen, hiddenPerHead});
    auto v2 = g->tensor({bs, heads, seqlen, hiddenPerHead});

    auto q3 = g->tensor({bs * heads, seqlen, hiddenPerHead});
    auto k3 = g->tensor({bs * heads, seqlen, hiddenPerHead});
    auto v3 = g->tensor({bs * heads, seqlen, hiddenPerHead});

    auto prob = g->tensor({bs * heads, seqlen, 2 * w + 1});
    auto probSoftmax = g->tensor({bs * heads, seqlen, 2 * w + 1});
    auto attn = g->tensor({bs * heads, seqlen, hiddenPerHead});

    auto t00 = g->tensor({bs, seqlen, hidden});
    auto t01 = g->tensor({bs, seqlen, hidden});
    auto t02 = g->tensor({bs, seqlen, hidden});
    // auto t10 = g->tensor({bs, seqlen, hidden});
    auto t11 = g->tensor({bs, seqlen, hidden});
    auto t12 = g->tensor({bs, seqlen, hidden});
    auto output = g->tensor({bs, seqlen, featlen});

    g->matmul(i0, w0, q0, false, true);
    g->matmul(i0, w1, k0, false, true);
    g->matmul(i0, w2, v0, false, true);
    g->reshape(q0, q1);
    g->reshape(k0, k1);
    g->reshape(v0, v1);
    g->transpose(q1, q2, 0, {{0, -1}, 2, 1, 3}, 1);
    g->transpose(k1, k2, 0, {{0, -1}, 2, 1, 3}, 1);
    g->transpose(v1, v2, 0, {{0, -1}, 2, 1, 3}, 1);
    g->reshape(q2, q3);
    g->reshape(k2, k3);
    g->reshape(v2, v3);
    // Attention
    g->g2bmm(q3, k3, prob, w, d);
    g->softmax(prob, probSoftmax, 2);
    g->gbmml(probSoftmax, v3, attn, d);
    g->transpose(attn, t00, 0, {0, 1, {-1, 2}}, heads);

    // Feed forward
    g->matmul(t00, w3, t01, false, true, bias3);
    g->relu(t01, t02);
    g->matmul(t02, w4, t11, false, true, bias4);
    g->relu(t11, t12);
    g->add({t12, i0}, output);

    g->updateConnection();

    std::shared_ptr<tpm::SubGraph> graph, bestGraph;
    graph = std::make_shared<tpm::SubGraph>(g->getOperators());
    auto mutationEngine = make_shared<tpm::NMutator>();
    mutationEngine->setMaxDepth(5);
    tpm::SearchEngine searchEngine(mutationEngine);
    searchEngine.run(graph, bestGraph);
    dbg("bestGraph");
    bestGraph->print();
    tpm::CodeEngine codeEngine;
    codeEngine.genCode(bestGraph, "res.cu");

    const auto originalTime = searchEngine.getPerf(graph, true);
    const auto bestTime = searchEngine.getPerf(bestGraph, true);
    dbg(originalTime, bestTime);
    EXPECT_GE(originalTime, 45);
    EXPECT_LE(bestTime, 25);
}

TEST(Longformer, g2bmm_bs1_d1) {
    const int bs = 1, seqlen = 10000, w = 1000, featlen = 512, heads = 8, d = 1;
    const int hidden = featlen, hiddenPerHead = hidden / heads;
    auto g = new tpm::Graph();
    auto q3 = g->tensor({bs * heads, seqlen, hiddenPerHead});
    auto k3 = g->tensor({bs * heads, seqlen, hiddenPerHead});
    // auto v3 = g->tensor({bs * heads, seqlen, hiddenPerHead});
    auto prob = g->tensor({bs * heads, seqlen, 2 * w + 1});
    g->g2bmm(q3, k3, prob, w, d);
    // g->softmax(prob, probSoftmax, 2);
    // g->gbmml(probSoftmax, v3, attn, d);
    g->updateConnection();

    std::shared_ptr<tpm::SubGraph> graph, bestGraph;
    graph = std::make_shared<tpm::SubGraph>(g->getOperators());
    tpm::SearchEngine searchEngine(make_shared<tpm::NMutator>());
    searchEngine.run(graph, bestGraph);
    dbg("bestGraph");
    bestGraph->print();

    const auto originalTime = searchEngine.getPerf(graph, true);
    EXPECT_GE(originalTime, 500);
    const auto bestTime = searchEngine.getPerf(bestGraph, true);
    EXPECT_LE(bestTime, 400);
}

TEST(Longformer, g2bmm_bs1_d4) {
    const int bs = 1, seqlen = 10000, w = 1000, featlen = 512, heads = 8, d = 4;
    const int hidden = featlen, hiddenPerHead = hidden / heads;
    auto g = new tpm::Graph();
    auto q3 = g->tensor({bs * heads, seqlen, hiddenPerHead});
    auto k3 = g->tensor({bs * heads, seqlen, hiddenPerHead});
    // auto v3 = g->tensor({bs * heads, seqlen, hiddenPerHead});
    auto prob = g->tensor({bs * heads, seqlen, 2 * w + 1});
    g->g2bmm(q3, k3, prob, w, d);
    // g->softmax(prob, probSoftmax, 2);
    // g->gbmml(probSoftmax, v3, attn, d);
    g->updateConnection();

    std::shared_ptr<tpm::SubGraph> graph, bestGraph;
    graph = std::make_shared<tpm::SubGraph>(g->getOperators());
    tpm::SearchEngine searchEngine(make_shared<tpm::NMutator>());
    searchEngine.run(graph, bestGraph);
    dbg("bestGraph");
    bestGraph->print();

    const auto originalTime = searchEngine.getPerf(graph, true);
    EXPECT_GE(originalTime, 500);
    const auto bestTime = searchEngine.getPerf(bestGraph, true);
    EXPECT_LE(bestTime, 400);
}

TEST(Longformer, e2e_bs16) {
    const int bs = 16, seqlen = 10000, w = 1000, featlen = 512, heads = 8,
              d = 4;
    const int hidden = featlen, hiddenPerHead = hidden / heads;
    assert(hidden % heads == 0);
    auto g = new tpm::Graph();

    auto i0 = g->tensor({bs * seqlen, featlen});
    auto w0 = g->tensor({featlen, hidden});
    auto w1 = g->tensor({512, 512});
    auto w2 = g->tensor({512, 512});
    // Feed forward
    auto w3 = g->tensor({512, 512});
    auto bias3 = g->tensor({512});
    auto w4 = g->tensor({512, 512});
    auto bias4 = g->tensor({512});

    auto q0 = g->tensor({bs * seqlen, hidden});
    auto k0 = g->tensor({bs * seqlen, hidden});
    auto v0 = g->tensor({bs * seqlen, hidden});

    auto q1 = g->tensor({bs, seqlen, heads, hiddenPerHead});
    auto k1 = g->tensor({bs, seqlen, heads, hiddenPerHead});
    auto v1 = g->tensor({bs, seqlen, heads, hiddenPerHead});

    auto q2 = g->tensor({bs, heads, seqlen, hiddenPerHead});
    auto k2 = g->tensor({bs, heads, seqlen, hiddenPerHead});
    auto v2 = g->tensor({bs, heads, seqlen, hiddenPerHead});

    auto q3 = g->tensor({bs * heads, seqlen, hiddenPerHead});
    auto k3 = g->tensor({bs * heads, seqlen, hiddenPerHead});
    auto v3 = g->tensor({bs * heads, seqlen, hiddenPerHead});

    // TODO check  membound time
    auto prob = g->tensor({bs * heads, seqlen, 2 * w + 1});
    auto probSoftmax = g->tensor({bs * heads, seqlen, 2 * w + 1});
    auto attn = g->tensor({bs * heads, seqlen, hiddenPerHead});
    // auto attnReshape = g->tensor({bs, heads, seqlen, hiddenPerHead});

    auto t0 = g->tensor({bs * seqlen, hidden});
    auto t00 = g->tensor({bs * seqlen, hidden});
    auto t01 = g->tensor({bs * seqlen, hidden});
    auto t02 = g->tensor({bs * seqlen, hidden});
    auto t11 = g->tensor({bs * seqlen, hidden});
    auto t12 = g->tensor({bs * seqlen, hidden});
    auto t13 = g->tensor({bs * seqlen, hidden});
    auto output = g->tensor({bs, seqlen, featlen});

    g->matmul(i0, w0, q0, false, true);
    g->matmul(i0, w1, k0, false, true);
    g->matmul(i0, w2, v0, false, true);
    g->reshape(q0, q1);
    g->reshape(k0, k1);
    g->reshape(v0, v1);
    g->transpose(q1, q2, 0, {{0, -1}, 2, 1, 3}, 1);
    g->transpose(k1, k2, 0, {{0, -1}, 2, 1, 3}, 1);
    g->transpose(v1, v2, 0, {{0, -1}, 2, 1, 3}, 1);
    g->reshape(q2, q3);
    g->reshape(k2, k3);
    g->reshape(v2, v3);
    // Attention
    g->g2bmm(q3, k3, prob, w, d);
    g->softmax(prob, probSoftmax, 2);
    g->gbmml(probSoftmax, v3, attn, d);
    // g->reshape(attn, attnReshape);
    // HOW TO DO IT
    // auto attn = g->tensor({bs * heads, seqlen, hiddenPerHead});
    // auto attnReshape = g->tensor({bs, heads, seqlen, hiddenPerHead});
    // auto t00 = g->tensor({bs, seqlen, heads*hiddenPerHeadidden});
    g->transpose(attn, t0, 0, {0, 1, {-1, 2}}, heads);
    // g->transpose(attnReshape, t00, 0, {0, -1, 2, 1, 3}, 1);
    g->reshape(t0, t00);

    // Feed forward
    g->matmul(t00, w3, t01, false, true, bias3);
    g->relu(t01, t02);
    g->matmul(t02, w4, t11, false, true, bias4);
    g->relu(t11, t12);
    g->add({t12, i0}, t13);
    g->reshape(t13, output);

    g->updateConnection();

    std::shared_ptr<tpm::SubGraph> graph, bestGraph;
    graph = std::make_shared<tpm::SubGraph>(g->getOperators());
    tpm::SearchEngine searchEngine(make_shared<tpm::NMutator>());
    searchEngine.run(graph, bestGraph);
    dbg("bestGraph");
    bestGraph->print();
    tpm::CodeEngine codeEngine;
    codeEngine.genCode(bestGraph, "res.cu");

    const auto originalTime = searchEngine.getPerf(graph, true);
    const auto bestTime = searchEngine.getPerf(bestGraph, true);
    dbg(originalTime, bestTime);
    EXPECT_GE(originalTime, 700);
    EXPECT_LE(bestTime, 400);
}